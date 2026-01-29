import torch.nn as nn

# Only needed for transfer learning models
from torchvision.models import resnet18, ResNet18_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNNv2(nn.Module):
    def __init__(self, num_classes: int = 10, dropout=0.25):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.0),
            ConvBlock(32, 64, dropout=dropout),
            ConvBlock(64, 128, dropout=dropout),
            ConvBlock(128, 256, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model(
    num_classes: int = 10,
    model_name: str = "simple_cnn_v2",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.25,
):
    model_name = (model_name or "simple_cnn_v2").lower()

    if model_name in ["simple_cnn_v2", "simplecnn", "cnn"]:
        return SimpleCNNv2(num_classes=num_classes, dropout=dropout)

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)

        # Replace classifier head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True

        return model

    raise ValueError(f"Unknown model_name: {model_name}")