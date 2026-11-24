"""
model.py
Model builder theo kiến trúc kế thừa.

Hierarchy:
    BaseClassifier        (ABC)
    └── ResNetClassifier  – pretrained ResNet với custom FC
        └── ResNet18Classifier  – cụ thể cho ResNet-18
    MultimodalClassifier  – Late Fusion: RGB (ResNet18) + MS (CNN) + HS (CNN)
"""

from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
import torch.nn as nn

from src.models import get_model_builder, get_rgb_backbone_spec

# ──────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────
class BaseClassifier(ABC, nn.Module):
    """
    Base class cho mọi classifier trong project.
    Subclass bắt buộc implement: build_backbone()
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.backbone    = self.build_backbone()

    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Trả về backbone đã gắn FC head phù hợp."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        print(f"[{self.__class__.__name__}]")
        print(f"  num_classes : {self.num_classes}")
        print(f"  parameters  : {self.count_parameters():,}")
        print(f"  device      : {next(self.parameters()).device}")


# ──────────────────────────────────────────────
# Generic ResNet classifier
# ──────────────────────────────────────────────
class ResNetClassifier(BaseClassifier):
    """
    Wrapper cho bất kỳ variant ResNet nào.
    Thay thế FC cuối bằng Linear(in_features, num_classes).
    """

    def __init__(
        self,
        num_classes: int,
        model_name:  str  = "resnet18",
        pretrained:  bool = True,
        dropout_p:   float = 0.0,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.dropout_p  = dropout_p
        super().__init__(num_classes)   # gọi build_backbone() bên trong

    def build_backbone(self) -> nn.Module:
        spec = get_rgb_backbone_spec(self.model_name)
        net = spec.model_fn(weights=spec.weights if self.pretrained else None)

        in_features = net.fc.in_features
        if self.dropout_p > 0.0:
            net.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_p),
                nn.Linear(in_features, self.num_classes),
            )
        else:
            net.fc = nn.Linear(in_features, self.num_classes)

        return net

    def freeze_backbone(self):
        """Đóng băng toàn bộ backbone, chỉ train FC head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("[Model] Backbone frozen – only FC head will be trained.")

    def unfreeze_all(self):
        """Mở đóng băng toàn bộ tham số (fine-tune)."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All parameters unfrozen.")


# ──────────────────────────────────────────────
# Concrete: ResNet-18 (dùng trong baseline)
# ──────────────────────────────────────────────
class ResNet18Classifier(ResNetClassifier):
    """
    ResNet-18 cụ thể – shortcut tiện dùng.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained:  bool  = True,
        dropout_p:   float = 0.0,
    ):
        super().__init__(
            num_classes=num_classes,
            model_name="resnet18",
            pretrained=pretrained,
            dropout_p=dropout_p,
        )


# ──────────────────────────────────────────────
# Factory function (tiện dùng từ config)
# ──────────────────────────────────────────────
def build_model(
    cfg:        Any,
    device:     torch.device,
    pretrained: bool  = True,
    dropout_p:  float = 0.0,
) -> ResNetClassifier:
    """
    Tạo model ResNet (single-modality RGB) từ ConfigDict và đưa lên device.

    Returns:
        model đã .to(device)
    """
    model_builder = get_model_builder(cfg.MODEL_NAME)
    if model_builder is not None:
        model = model_builder(
            num_classes=cfg.NUM_CLASSES,
            pretrained=pretrained,
            dropout=dropout_p,
        ).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{model.__class__.__name__}]")
        print(f"  num_classes : {cfg.NUM_CLASSES}")
        print(f"  pretrained  : {pretrained}")
        print(f"  dropout     : {dropout_p}")
        print(f"  parameters  : {total_params:,}")
        print(f"  device      : {next(model.parameters()).device}")
        return model

    model = ResNetClassifier(
        num_classes=cfg.NUM_CLASSES,
        model_name=cfg.MODEL_NAME,
        pretrained=pretrained,
        dropout_p=dropout_p,
    ).to(device)

    model.summary()
    return model


# ──────────────────────────────────────────────
# MS Branch: CNN nhỏ cho 5 kênh đầu vào
# ──────────────────────────────────────────────
class MSBranch(nn.Module):
    """
    Lightweight CNN cho ảnh Multispectral (5 channels, 64x64).
    Output: feature vector 256 chiều.
    """

    def __init__(self, out_features: int = 256, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 5 → 32 channels, 64x64 → 32x32
            nn.Conv2d(5, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64 channels, 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128 channels, 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 → 256 channels, 8x8 → 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # → (256, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# ──────────────────────────────────────────────
# HS Branch: CNN cho 125 kênh đầu vào (32x32)
# ──────────────────────────────────────────────
class HSBranch(nn.Module):
    """
    CNN cho ảnh Hyperspectral (125 channels, 32x32).
    Dùng Conv2d với in_channels=125; pooling aggressively vì spatial nhỏ.
    Output: feature vector 256 chiều.
    """

    def __init__(self, out_features: int = 256, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 125 → 256 channels, 32x32 → 16x16
            # Dùng kernel 1x1 trước để giảm kênh (spectral mixing)
            nn.Conv2d(125, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 2: 256 → 256, spatial 32x32 → 16x16
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 256 → 256, 16x16 → 8x8
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global average pool → (256, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# ──────────────────────────────────────────────
# MultimodalClassifier – Late Fusion
# ──────────────────────────────────────────────
class MultimodalClassifier(nn.Module):
    """
    Late Fusion model cho 3 modality: RGB, MS, HS.

    Architecture:
        RGB Branch  → ResNet18 (pretrained, FC removed) → 512-d
        MS Branch   → MSBranch (5 ch CNN)               → 256-d
        HS Branch   → HSBranch (125 ch CNN)              → 256-d
        Fusion Head → Concat(512+256+256=1024) → FC → num_classes

    Args:
        num_classes:     Số class phân loại.
        rgb_pretrained:  Sử dụng pretrained weights cho RGB branch.
        rgb_model:       Tên ResNet variant ('resnet18', 'resnet34', ...).
        ms_out:          Output dim của MS branch (default 256).
        hs_out:          Output dim của HS branch (default 256).
        dropout_p:       Dropout probability ở Fusion Head & branches.
        fusion_hidden:   Số neurons lớp ẩn trong Fusion Head (0 = bỏ qua).
    """

    def __init__(
        self,
        num_classes:    int,
        rgb_pretrained: bool  = True,
        rgb_model:      str   = "resnet18",
        ms_out:         int   = 256,
        hs_out:         int   = 256,
        dropout_p:      float = 0.3,
        fusion_hidden:  int   = 512,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── RGB Branch (ResNet backbone, bỏ FC cuối) ──────────────────────────
        spec = get_rgb_backbone_spec(rgb_model)
        rgb_feat = spec.out_features
        _resnet = spec.model_fn(weights=spec.weights if rgb_pretrained else None)
        # Bỏ lớp FC cuối → chỉ giữ feature extractor
        self.rgb_branch = nn.Sequential(*list(_resnet.children())[:-1])  # → (B, rgb_feat, 1, 1)
        self.rgb_flatten = nn.Flatten()

        # ── MS & HS Branches ─────────────────────────────────────────────────
        self.ms_branch = MSBranch(out_features=ms_out, dropout_p=dropout_p)
        self.hs_branch = HSBranch(out_features=hs_out, dropout_p=dropout_p)

        # ── Fusion Head ────────────────────────────────────────────────────────
        fused_dim = rgb_feat + ms_out + hs_out   # 512 + 256 + 256 = 1024
        if fusion_hidden > 0:
            self.fusion_head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(fused_dim, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(fusion_hidden, num_classes),
            )
        else:
            self.fusion_head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(fused_dim, num_classes),
            )

    def forward(
        self,
        hs:  torch.Tensor,   # (B, 125, 32, 32)
        ms:  torch.Tensor,   # (B,   5, 64, 64)
        rgb: torch.Tensor,   # (B,   3, H,  W)
    ) -> torch.Tensor:       # (B, num_classes)
        rgb_feat = self.rgb_flatten(self.rgb_branch(rgb))  # (B, 512)
        ms_feat  = self.ms_branch(ms)                       # (B, 256)
        hs_feat  = self.hs_branch(hs)                       # (B, 256)

        fused = torch.cat([rgb_feat, ms_feat, hs_feat], dim=1)  # (B, 1024)
        return self.fusion_head(fused)                           # (B, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        total = self.count_parameters()
        rgb_p  = sum(p.numel() for p in self.rgb_branch.parameters())
        ms_p   = sum(p.numel() for p in self.ms_branch.parameters())
        hs_p   = sum(p.numel() for p in self.hs_branch.parameters())
        head_p = sum(p.numel() for p in self.fusion_head.parameters())
        print("[MultimodalClassifier]")
        print(f"  num_classes   : {self.num_classes}")
        print(f"  total params  : {total:,}")
        print(f"    rgb_branch  : {rgb_p:,}")
        print(f"    ms_branch   : {ms_p:,}")
        print(f"    hs_branch   : {hs_p:,}")
        print(f"    fusion_head : {head_p:,}")
        print(f"  device        : {next(self.parameters()).device}")

    def freeze_rgb_branch(self):
        """Đóng băng RGB branch để chỉ fine-tune MS, HS và Fusion Head."""
        for param in self.rgb_branch.parameters():
            param.requires_grad = False
        print("[MultimodalClassifier] RGB branch frozen.")

    def unfreeze_all(self):
        """Mở đóng băng toàn bộ tham số."""
        for param in self.parameters():
            param.requires_grad = True
        print("[MultimodalClassifier] All parameters unfrozen.")


# ──────────────────────────────────────────────
# Spectral Attention: SE Block chọn lọc bước sóng
# ──────────────────────────────────────────────
class SpectralAttention(nn.Module):
    """Squeeze-and-Excitation trên chiều kênh phổ (125 kênh HS)."""
    def __init__(self, channels: int = 125, reduction: int = 8):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class CompactHSBranch(nn.Module):
    """Nhánh trích xuất đặc trưng Siêu phổ nhỏ gọn kèm Spectral Attention (ra 128-d)."""
    def __init__(self, in_channels: int = 125, out_features: int = 128, dropout_p: float = 0.3):
        super().__init__()
        self.attention = SpectralAttention(channels=in_channels, reduction=8)
        self.net = nn.Sequential(
            # Giảm chiều kênh phổ từ 125 xuống 64
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Trích xuất đặc trưng không gian
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_att = self.attention(x)
        return self.head(self.net(x_att))


class CompactMSBranch(nn.Module):
    """Nhánh trích xuất đặc trưng Đa phổ nhỏ gọn (ra 128-d)."""
    def __init__(self, in_channels: int = 5, out_features: int = 128, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class SpectralGatedFusionClassifier(nn.Module):
    """
    SG-CFNet: Spectral-Gated Compact Fusion Network.
    - Cân bằng chiều đặc trưng (mỗi nhánh 128 chiều).
    - Chuẩn hóa L2-Norm để các nhánh không lấn át lẫn nhau.
    - Cổng Gated Multimodal Attention học trọng số động (alpha_rgb, alpha_ms, alpha_hs).
    - Modality Dropout (15% cơ hội tắt bớt một kênh trong training).
    """
    def __init__(
        self,
        num_classes: int = 3,
        rgb_model: str = "resnet18",
        rgb_pretrained: bool = True,
        branch_dim: int = 128,
        dropout_p: float = 0.3,
        modality_dropout: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.branch_dim = branch_dim
        self.modality_dropout = modality_dropout

        # 1. Nhánh RGB Pretrained
        spec = get_rgb_backbone_spec(rgb_model)
        _resnet = spec.model_fn(weights=spec.weights if rgb_pretrained else None)
        self.rgb_branch = nn.Sequential(*list(_resnet.children())[:-1])
        self.rgb_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(spec.out_features, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Nhánh MS & HS
        self.ms_branch = CompactMSBranch(in_channels=5, out_features=branch_dim, dropout_p=dropout_p)
        self.hs_branch = CompactHSBranch(in_channels=125, out_features=branch_dim, dropout_p=dropout_p)

        # 3. Cổng Gated Attention (tính alpha cho 3 modality)
        self.gate_fc = nn.Sequential(
            nn.Linear(branch_dim * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
            nn.Softmax(dim=1),
        )

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(branch_dim * 2, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(branch_dim, num_classes),
        )

    def forward(self, hs: torch.Tensor, ms: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_proj(self.rgb_branch(rgb))
        f_ms  = self.ms_branch(ms)
        f_hs  = self.hs_branch(hs)

        # Chuẩn hóa L2-Normalization
        f_rgb_norm = nn.functional.normalize(f_rgb, p=2, dim=1)
        f_ms_norm  = nn.functional.normalize(f_ms,  p=2, dim=1)
        f_hs_norm  = nn.functional.normalize(f_hs,  p=2, dim=1)

        # Modality dropout trong lúc train
        if self.training and self.modality_dropout > 0:
            mask = torch.rand(f_rgb.size(0), 2, device=f_rgb.device) > self.modality_dropout
            f_ms_norm = f_ms_norm * mask[:, 0:1]
            f_hs_norm = f_hs_norm * mask[:, 1:2]

        # Ghép đặc trưng và tính cổng Gated weights (B, 3)
        concat_all = torch.cat([f_rgb_norm, f_ms_norm, f_hs_norm], dim=1)
        gates = self.gate_fc(concat_all)

        # Gated Weighted Sum kết hợp với Concatenated Gated features
        gated_fused = (
            gates[:, 0:1] * f_rgb_norm +
            gates[:, 1:2] * f_ms_norm +
            gates[:, 2:3] * f_hs_norm
        )
        
        # Kết hợp thông tin tổng hợp và trọng số RGB
        combined = torch.cat([gated_fused, f_rgb_norm], dim=1)
        return self.classifier(combined)

    def freeze_rgb_branch(self):
        """Đóng băng nhánh RGB để khởi động và huấn luyện trước hai nhánh phổ HS & MS."""
        for param in self.rgb_branch.parameters():
            param.requires_grad = False
        print("[SG-CFNet] RGB branch frozen for warm-up.")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("[SG-CFNet] All parameters unfrozen for joint fine-tuning.")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        total = self.count_parameters()
        rgb_p  = sum(p.numel() for p in self.rgb_branch.parameters()) + sum(p.numel() for p in self.rgb_proj.parameters())
        ms_p   = sum(p.numel() for p in self.ms_branch.parameters())
        hs_p   = sum(p.numel() for p in self.hs_branch.parameters())
        gate_p = sum(p.numel() for p in self.gate_fc.parameters())
        head_p = sum(p.numel() for p in self.classifier.parameters())
        print("[SpectralGatedFusionClassifier (SG-CFNet)]")
        print(f"  num_classes   : {self.num_classes}")
        print(f"  total params  : {total:,}")
        print(f"    rgb_branch  : {rgb_p:,}")
        print(f"    ms_branch   : {ms_p:,}")
        print(f"    hs_branch   : {hs_p:,}")
        print(f"    gate_fc     : {gate_p:,}")
        print(f"    classifier  : {head_p:,}")
        print(f"  device        : {next(self.parameters()).device}")


# ──────────────────────────────────────────────
# SS-CFNet v5: Spectral-Spatial Compact Fusion Network
# ──────────────────────────────────────────────
class PixelSNV(nn.Module):
    """
    Standard Normal Variate (SNV) theo từng pixel dọc theo chiều kênh phổ (dim=1).
    Khử hoàn toàn biến thiên chiếu sáng, bóng đổ do độ cong của lá và viền đen.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        valid_mask = (std > 1e-4).float()
        return (x - mean) / (std + self.eps) * valid_mask


class AdvancedHSBranch(nn.Module):
    """
    Nhánh Siêu phổ nâng cao:
    1. Per-pixel SNV: chuẩn hóa phổ triệt tiêu bóng đổ / chiếu sáng.
    2. Đạo hàm phổ bậc 1 (Spectral Derivative): nắm bắt độ dốc Red-Edge (bước nhảy 690-740nm).
    3. Ghép nối 125 kênh SNV + 124 kênh Derivative = 249 kênh.
    4. 1x1 Conv giảm số kênh xuống 64, theo sau bởi 2 khối tích chập không gian.
    5. Bộ bảo vệ tensor lỗi (zero-tensor guard): tự động bỏ qua nếu ảnh HS rỗng.
    """
    def __init__(self, in_channels: int = 125, out_features: int = 128, dropout_p: float = 0.3):
        super().__init__()
        self.snv = PixelSNV()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels + (in_channels - 1), 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(128, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        is_sample_valid = (hs.abs().sum(dim=[1, 2, 3], keepdim=True) > 1e-3).float()
        hs_snv = self.snv(hs)
        d1 = hs_snv[:, 1:] - hs_snv[:, :-1]
        cube = torch.cat([hs_snv, d1], dim=1)
        feat = self.conv1x1(cube)
        feat = self.conv_blocks(feat)
        out = self.fc(feat)
        return out * is_sample_valid.squeeze(-1).squeeze(-1)


class AdvancedMSBranch(nn.Module):
    """
    Nhánh Đa phổ nâng cao:
    Tích hợp trực tiếp các chỉ số thực vật hóa sinh:
    - NDVI: (NIR - Red) / (NIR + Red)
    - NDRE: (NIR - RedEdge) / (NIR + RedEdge)
    - GNDVI: (NIR - Green) / (NIR + Green)
    Ghép 5 kênh phổ gốc + 3 kênh chỉ số = 8 kênh đầu vào.
    """
    def __init__(self, in_channels: int = 5, out_features: int = 64, dropout_p: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(64, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
        )

    def forward(self, ms: torch.Tensor) -> torch.Tensor:
        is_sample_valid = (ms.abs().sum(dim=[1, 2, 3], keepdim=True) > 1e-3).float()
        eps = 1e-5
        ndvi = (ms[:, 4:5] - ms[:, 2:3]) / (ms[:, 4:5] + ms[:, 2:3] + eps)
        ndre = (ms[:, 4:5] - ms[:, 3:4]) / (ms[:, 4:5] + ms[:, 3:4] + eps)
        gndvi = (ms[:, 4:5] - ms[:, 1:2]) / (ms[:, 4:5] + ms[:, 1:2] + eps)

        ms_aug = torch.cat([ms, ndvi, ndre, gndvi], dim=1)
        feat = self.conv(ms_aug)
        out = self.fc(feat)
        return out * is_sample_valid.squeeze(-1).squeeze(-1)


class SSCFNetClassifier(nn.Module):
    """
    Spectral-Spatial Compact Fusion Network (SS-CFNet v5).
    Kiến trúc Residual Multimodal Fusion kết hợp ưu thế:
    1. RGB Branch (ResNet-18 ImageNet pretrained) giữ nguyên 512 đặc trưng và đưa trực tiếp vào RGB Logits.
       -> Bảo vệ 82.5% độ nhận diện Health & Other, không bị ép hẹp bottleneck.
    2. Spectral Branch:
       - HS Branch (128-d): Pixel-SNV + Spectral Derivative (Red-Edge).
       - MS Branch (64-d): NDVI, NDRE, GNDVI.
       -> Spectral Head đưa ra Refinement Logits chuyên biệt để phát hiện Rust ẩn.
    3. Cổng điều tiết động Adaptive Residual Gate (alpha):
       alpha = Sigmoid(Linear(RGB + Spectral)) in [0, 1].
       Logits = Logits_RGB + alpha * Logits_Spectral.
    """
    def __init__(
        self,
        num_classes: int = 3,
        rgb_model: str = "resnet18",
        rgb_pretrained: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 1. RGB Primary Branch
        spec = get_rgb_backbone_spec(rgb_model)
        _resnet = spec.model_fn(weights=spec.weights if rgb_pretrained else None)
        self.rgb_branch = nn.Sequential(*list(_resnet.children())[:-1])
        self.rgb_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(spec.out_features, num_classes),
        )

        # 2. Spectral Refinement Branches
        self.hs_branch = AdvancedHSBranch(in_channels=125, out_features=128, dropout_p=dropout_p)
        self.ms_branch = AdvancedMSBranch(in_channels=5, out_features=64, dropout_p=dropout_p)

        self.spectral_head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

        # 3. Dynamic Residual Gate
        self.gate_fc = nn.Sequential(
            nn.Linear(spec.out_features + 192, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, hs: torch.Tensor, ms: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        f_rgb = torch.flatten(self.rgb_branch(rgb), 1)
        logits_rgb = self.rgb_head(f_rgb)

        f_hs = self.hs_branch(hs)
        f_ms = self.ms_branch(ms)
        f_spectral = torch.cat([f_hs, f_ms], dim=1)
        logits_spectral = self.spectral_head(f_spectral)

        alpha = self.gate_fc(torch.cat([f_rgb, f_spectral], dim=1))
        return logits_rgb + alpha * logits_spectral

    def freeze_rgb_branch(self):
        for param in self.rgb_branch.parameters():
            param.requires_grad = False
        print("[SS-CFNet] RGB branch frozen for warm-up.")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        print("[SS-CFNet] All parameters unfrozen for fine-tuning.")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        total = self.count_parameters()
        rgb_p = sum(p.numel() for p in self.rgb_branch.parameters()) + sum(p.numel() for p in self.rgb_head.parameters())
        hs_p  = sum(p.numel() for p in self.hs_branch.parameters())
        ms_p  = sum(p.numel() for p in self.ms_branch.parameters())
        spec_head_p = sum(p.numel() for p in self.spectral_head.parameters())
        gate_p = sum(p.numel() for p in self.gate_fc.parameters())
        print("[SS-CFNet v5 Classifier]")
        print(f"  num_classes   : {self.num_classes}")
        print(f"  total params  : {total:,}")
        print(f"    rgb_branch  : {rgb_p:,}")
        print(f"    hs_branch   : {hs_p:,}")
        print(f"    ms_branch   : {ms_p:,}")
        print(f"    spectral_head: {spec_head_p:,}")
        print(f"    gate_fc     : {gate_p:,}")
        print(f"  device        : {next(self.parameters()).device}")


def build_multimodal_model(
    cfg:    Any,
    device: torch.device,
) -> nn.Module:
    """
    Tạo mô hình Multimodal từ ConfigDict và đưa lên device.
    Hỗ trợ cả 'ss_cfnet' (v5 Residual), 'spectral_gated' (SG-CFNet v1-4), và 'naive' Late Fusion.
    """
    fusion_type = getattr(cfg, 'FUSION_TYPE', 'ss_cfnet').lower()

    if fusion_type in {'ss_cfnet', 'sscfnet', 'spectral_spatial', 'residual', 'spectral_gated', 'sg_cfnet', 'gated'}:
        model = SSCFNetClassifier(
            num_classes    = cfg.NUM_CLASSES,
            rgb_model      = getattr(cfg, 'MODEL_NAME', 'resnet18'),
            rgb_pretrained = getattr(cfg, 'RGB_PRETRAINED', True),
            dropout_p      = getattr(cfg, 'DROPOUT_P', 0.3),
        ).to(device)
        model.summary()
        return model

    model = MultimodalClassifier(
        num_classes    = cfg.NUM_CLASSES,
        rgb_pretrained = getattr(cfg, 'RGB_PRETRAINED',  True),
        rgb_model      = getattr(cfg, 'MODEL_NAME',      'resnet18'),
        ms_out         = getattr(cfg, 'MS_OUT_FEATURES', 256),
        hs_out         = getattr(cfg, 'HS_OUT_FEATURES', 256),
        dropout_p      = getattr(cfg, 'DROPOUT_P',       0.3),
        fusion_hidden  = getattr(cfg, 'FUSION_HIDDEN',   512),
    ).to(device)

    model.summary()
    return model
