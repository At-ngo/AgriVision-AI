import torch
import torch.nn as nn
from torchvision import models

class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.convnext_tiny(weights=weights)
        
        # ConvNeXt classifier head is a Sequential containing LayerNorm2d, Flatten, Linear
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Đóng băng feature extractor, chỉ train classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("[Model] Backbone frozen – only classifier head will be trained.")

    def unfreeze_all(self):
        """Mở đóng băng toàn bộ tham số."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All parameters unfrozen.")

def main():
    model = ConvNeXtTiny(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)

if __name__ == "__main__":
    main()
