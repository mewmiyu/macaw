from torch import nn
from torchvision import models


class FeatureNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.Inception3(aux_logits=False,init_weights=True)
        inception_modules = list(inception.children())[:-1]
        self.backbone = nn.Sequential(*list(inception_modules))
        self.fc = nn.Sequential(nn.Linear(2048, 512))


    def forward(self, x):
        x = self.backbone(x)
        #Output is (B, C, H, W), but linear layer needs (B, H, W, C)
        x = x.permute(0,2,3,1)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2)
        return x
