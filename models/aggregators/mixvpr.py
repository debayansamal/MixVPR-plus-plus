import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=1,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()

        self.in_h = in_h # height of input feature maps
        self.in_w = in_w # width of input feature maps
        self.in_channels = in_channels # depth of input feature maps
        
        self.out_channels = out_channels # depth wise projection dimension
        self.out_rows = out_rows # row wise projection dimesion

        self.mix_depth = mix_depth # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio # ratio of the mid projection layer in the mixer block

        hw = in_h*in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class LearnableGaborFilter(nn.Module):
    """Depthwise texture filter initialized with a small Gabor filter bank."""

    def __init__(self, channels, kernel_size=7, num_orientations=4):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd for same padding')

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        base_kernels = self._build_gabor_bank(kernel_size, num_orientations)
        weight = base_kernels.repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(weight)
        self.scale = nn.Parameter(torch.ones(channels * num_orientations))

    @staticmethod
    def _build_gabor_bank(kernel_size, num_orientations):
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')

        sigma = kernel_size / 3.0
        wavelength = kernel_size / 2.0
        kernels = []
        for i in range(num_orientations):
            theta = np.pi * i / num_orientations
            x_theta = xx * np.cos(theta) + yy * np.sin(theta)
            y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
            envelope = torch.exp(-(x_theta ** 2 + y_theta ** 2) / (2 * sigma ** 2))
            carrier = torch.cos(2 * np.pi * x_theta / wavelength)
            kernel = envelope * carrier
            kernel = kernel - kernel.mean()
            kernel = kernel / (kernel.norm(p=2) + 1e-6)
            kernels.append(kernel)

        return torch.stack(kernels).unsqueeze(1)

    def forward(self, x):
        texture = F.conv2d(
            x,
            self.weight * self.scale.view(-1, 1, 1, 1),
            padding=self.padding,
            groups=self.channels,
        )
        return texture


class AdaptiveGaborTextureFuser(nn.Module):
    def __init__(self, channels, kernel_size=7, num_orientations=4):
        super().__init__()
        self.gabor = LearnableGaborFilter(channels, kernel_size, num_orientations)
        hidden_channels = channels * num_orientations
        self.texture_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        texture = self.texture_proj(self.gabor(x))
        gate = self.gate(torch.cat([x, texture], dim=1))
        return x + gate * texture


class HierarchicalRegionFeatureMixer(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
        region_scales=(1, 2, 4),
    ):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_rows = out_rows
        self.region_scales = tuple(region_scales)

        self.global_mixer = nn.Sequential(*[
            FeatureMixerLayer(in_dim=in_h * in_w, mlp_ratio=mlp_ratio)
            for _ in range(mix_depth)
        ])
        self.region_mixers = nn.ModuleList([
            FeatureMixerLayer(in_dim=scale * scale, mlp_ratio=mlp_ratio)
            for scale in self.region_scales
            if scale > 1
        ])

        num_branches = 1 + len(self.region_mixers)
        self.branch_fuse = nn.Sequential(
            nn.LayerNorm(in_channels * num_branches),
            nn.Linear(in_channels * num_branches, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels),
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(in_h * in_w, out_rows)

    def _mix_regions(self, x, mixer, scale):
        pooled = F.adaptive_avg_pool2d(x, output_size=(scale, scale))
        pooled = pooled.flatten(2)
        pooled = mixer(pooled)
        pooled = pooled.view(x.shape[0], x.shape[1], scale, scale)
        return F.interpolate(pooled, size=(self.in_h, self.in_w), mode='bilinear', align_corners=False)

    def forward(self, x):
        branches = []
        global_features = x.flatten(2)
        global_features = self.global_mixer(global_features)
        branches.append(global_features)

        mixer_idx = 0
        for scale in self.region_scales:
            if scale <= 1:
                continue
            region_features = self._mix_regions(x, self.region_mixers[mixer_idx], scale)
            branches.append(region_features.flatten(2))
            mixer_idx += 1

        x = torch.cat(branches, dim=1).permute(0, 2, 1)
        x = self.branch_fuse(x)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x


class MixVPRPlusPlus(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=1,
        mlp_ratio=1,
        out_rows=4,
        region_scales=(1, 2, 4),
        gabor_kernel_size=7,
        gabor_orientations=4,
    ) -> None:
        super().__init__()
        self.texture_fuser = AdaptiveGaborTextureFuser(
            in_channels,
            kernel_size=gabor_kernel_size,
            num_orientations=gabor_orientations,
        )
        self.region_mixer = HierarchicalRegionFeatureMixer(
            in_channels=in_channels,
            in_h=in_h,
            in_w=in_w,
            out_channels=out_channels,
            mix_depth=mix_depth,
            mlp_ratio=mlp_ratio,
            out_rows=out_rows,
            region_scales=region_scales,
        )

    def forward(self, x):
        x = self.texture_fuser(x)
        return self.region_mixer(x)


# -------------------------------------------------------------------------------

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')


def main():
    x = torch.randn(1, 1024, 20, 20)
    agg = MixVPR(
        in_channels=1024,
        in_h=20,
        in_w=20,
        out_channels=1024,
        mix_depth=4,
        mlp_ratio=1,
        out_rows=4)

    print_nb_params(agg)
    output = agg(x)
    print(output.shape)


if __name__ == '__main__':
    main()
