import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HybridColorModel(nn.Module):
    """
    Görsel + renk vektörünü birleştirip uyum skoru ve/veya sınıfı üreten hibrit model.
    """

    def __init__(self, color_dim: int = 15, backbone_name: str = "resnet50",
                 output_regression: bool = True, output_classification: bool = True):
        super().__init__()

        self.output_regression = output_regression
        self.output_classification = output_classification

        # Backbone
        backbone = getattr(models, backbone_name)(pretrained=True)
        # Son classification katmanını kaldır
        if hasattr(backbone, "fc"):
            backbone_out_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError("Desteklenmeyen backbone: {0}".format(backbone_name))

        self.backbone = backbone

        # Renk embedding katmanı
        self.color_mlp = nn.Sequential(
            nn.Linear(color_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        fusion_dim = backbone_out_dim + 64

        # Ortak gövde
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        if self.output_regression:
            self.reg_head = nn.Linear(256, 1)  # uyum skoru (0-1 veya 0-100)

        if self.output_classification:
            self.clf_head = nn.Linear(256, 2)  # uyumlu / uyumsuz

    def forward(self, image, color_vec):
        # image: (B, C, H, W)
        # color_vec: (B, color_dim)

        img_feat = self.backbone(image)
        color_feat = self.color_mlp(color_vec)

        fused = torch.cat([img_feat, color_feat], dim=1)
        fused = self.fusion(fused)

        outputs = {}
        if self.output_regression:
            reg = self.reg_head(fused)
            outputs["score"] = reg

        if self.output_classification:
            logits = self.clf_head(fused)
            outputs["logits"] = logits

        return outputs
