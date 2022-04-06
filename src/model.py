from torch import nn
import timm


class ChulaFoodNet(nn.Module):
    def __init__(self, num_classes):
        super(ChulaFoodNet, self).__init__()
        self.pretrained_model = timm.create_model(
            "convnext_base_in22k", pretrained=True, drop_rate=0.2
        )
        self.pretrained_model.head.fc = nn.Linear(1024, num_classes)

    def forward(self, input):
        x = self.pretrained_model(input)
        return x
