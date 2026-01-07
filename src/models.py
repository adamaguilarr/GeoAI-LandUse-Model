import torch.nn as nn
import torchvision.models as models


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
    num_classes: int,
    model_name: str = "cnn_v2",
    dropout: float = 0.25,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    if model_name == "cnn_v2":
        return SimpleCNNv2(num_classes=num_classes, dropout=dropout)

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)

        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False

        return model

    raise ValueError(f"Unknown model_name: {model_name}")