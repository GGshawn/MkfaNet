import sys
sys.path.append('/home/gc/projects/DeepfakeBench/')

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from metrics.registry import BACKBONE
# from training.metrics.registry import BACKBONE

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class MKA(nn.Module):
    """Multi-Kernel Aggregation"""
    def __init__(self, channels):
        super().__init__()
        k = channels // 3
        self.slices = (k, k, channels - 2 * k)
        k1, k2, k3 = self.slices

        self.norm = nn.BatchNorm2d(channels)
        self.dw1 = nn.Conv2d(k1, k1, 7, padding=3, dilation=1, groups=k1, bias=False)
        self.dw2 = nn.Conv2d(k2, k2, 7, padding=6, dilation=2, groups=k2, bias=False)
        self.dw3 = nn.Conv2d(k3, k3, 7, padding=9, dilation=3, groups=k3, bias=False)
        self.g_x = nn.Conv2d(channels, channels, 1, bias=True)
        self.g_yc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x):
        idn = x
        x = self.norm(x)
        k1, k2, k3 = self.slices
        xl, xm, xh = torch.split(x, [k1, k2, k3], dim=1)
        yl = self.dw1(xl)
        ym = self.dw2(xm)
        yh = self.dw3(xh)
        yc = torch.cat([yl, ym, yh], dim=1)
        out = self.act(self.g_x(x)) * self.act(self.g_yc(yc))
        return idn + out


class MFA(nn.Module):
    def __init__(self, channels, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(channels * mlp_ratio)
        self.norm = nn.BatchNorm2d(channels)
        # 1×1 扩张 + 3×3 depthwise 表征 + GELU
        self.fc1 = nn.Conv2d(channels, hidden_dim, 1)
        self.dw = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=True)
        self.act = nn.GELU()
        # 频域增强参数
        self.gamma = nn.Parameter(torch.zeros(1, hidden_dim, 1, 1))
        # 压缩回原通道
        self.fc2 = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x):
        idn = x
        x = self.norm(x)
        y = self.act(self.dw(self.fc1(x)))  # [B, hidden_dim, H, W]
        # === 频域增强 ===
        z_dc = F.adaptive_avg_pool2d(y, 1)
        z_l = y.mean(dim=1, keepdim=True)
        y_dc = z_dc * y
        y_hc = y - z_l * y
        y_mf = y_dc + self.gamma * y_hc
        return idn + self.fc2(y_mf)


class MkfaBlock(nn.Module):
    def __init__(self, channels, mlp_ratio=4):
        super().__init__()
        self.mka = MKA(channels)
        self.mfa = MFA(channels, mlp_ratio)

    def forward(self, x):
        x = self.mka(x)
        x = self.mfa(x)
        return x


class StackConvPatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, 2, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.stem(x)


class ConvPatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.proj(x))



@BACKBONE.register_module(module_name="mkfanet")
class MkfaNet(nn.Module):
    """
    MkfaNet backbone (tiny/small/base)
    """
    ARCHS = {
        "tiny":  dict(embed=[32, 64, 128, 256], depths=[3, 3, 12, 2], mlp=[8, 8, 4, 4]),
        "small":  dict(embed=[64, 128, 320, 512], depths=[2, 3, 10, 2], mlp=[8, 8, 4, 4]),
    }

    def __init__(self, mkfanet_config):
        super(MkfaNet, self).__init__()
        self.num_classes = mkfanet_config["num_classes"]
        self.mode = mkfanet_config.get("mode", "default")
        self.inc = mkfanet_config.get("inc", 3)
        self.dropout = mkfanet_config.get("dropout", 0.0)
        arch = mkfanet_config.get("arch", "base")

        assert arch in self.ARCHS, f"Unsupported arch {arch}, choose from {list(self.ARCHS.keys())}"
        embed_dims = self.ARCHS[arch]["embed"]
        depths = self.ARCHS[arch]["depths"]
        mlp_ratios = self.ARCHS[arch]["mlp"]

        # Stages
        self.patch_embed1 = StackConvPatchEmbed(self.inc, embed_dims[0])
        self.blocks1 = nn.Sequential(*[MkfaBlock(embed_dims[0], mlp_ratios[0]) for _ in range(depths[0])])

        self.patch_embed2 = ConvPatchEmbed(embed_dims[0], embed_dims[1])
        self.blocks2 = nn.Sequential(*[MkfaBlock(embed_dims[1], mlp_ratios[1]) for _ in range(depths[1])])

        self.patch_embed3 = ConvPatchEmbed(embed_dims[1], embed_dims[2])
        self.blocks3 = nn.Sequential(*[MkfaBlock(embed_dims[2], mlp_ratios[2]) for _ in range(depths[2])])

        self.patch_embed4 = ConvPatchEmbed(embed_dims[2], embed_dims[3])
        self.blocks4 = nn.Sequential(*[MkfaBlock(embed_dims[3], mlp_ratios[3]) for _ in range(depths[3])])

        # Head
        final_channel = embed_dims[-1]
        if self.mode == "adjust_channel_iid":
            final_channel = 512
            self.mode = "adjust_channel"

        if self.dropout:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(final_channel, self.num_classes)
            )
        else:
            self.last_linear = nn.Linear(final_channel, self.num_classes)

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )
        self.relu = nn.ReLU(inplace=True)

    def fea_part1(self, x):
        x = self.patch_embed1(x)
        x = self.blocks1(x)
        return x

    def fea_part2(self, x):
        x = self.patch_embed2(x)
        x = self.blocks2(x)
        return x

    def fea_part3(self, x):
        x = self.patch_embed3(x)
        x = self.blocks3(x)
        return x

    def fea_part4(self, x):
        x = self.patch_embed4(x)
        x = self.blocks4(x)
        return x

    def features(self, x):
        x = self.fea_part1(x)
        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, features, id_feat=None):
        x = features if self.mode == "adjust_channel" else self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        self.last_emb = x
        out = self.last_linear(x - id_feat) if id_feat is not None else self.last_linear(x)
        return out

    def forward(self, input):
        x = self.features(input)
        out = self.classifier(x)
        return out, x



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="tiny", choices=["tiny", "small"])
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--mode", type=str, default="default")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    cfg = {
        "num_classes": args.num_classes,
        "mode": args.mode,
        "inc": 3,
        "dropout": args.dropout,
        "arch": args.arch,
    }

    model = MkfaNet(cfg)
    x = torch.randn(1, 3, args.size, args.size)
    
    model.eval()
    x = torch.randn(1, 3, args.size, args.size)


    from torchinfo import summary

    summary(model, x.shape, device="cpu")


    
    out, feat = model(x)
    print(f"Arch={args.arch}")
    print("Output shape:", out.shape)
    print("Feature shape:", feat.shape)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.2f}M")

