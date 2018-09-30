import torch.nn as nn
from torchvision.models.resnet import ResNet

class MyBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality=32, stride=1, downsample=None):
        super(MyBottleneck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2),
            self.relu,
            nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                      padding=1, bias=False, groups=cardinality),
            nn.BatchNorm2d(planes * 2),
            self.relu,
            nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.downsample = downsample


    def forward(self, x):
        residual = x

        print(x.shape)
        out = self.conv(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet50(pretrained=False, **kwargs):
    model = ResNet(MyBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load(pretrained))
        model.eval()
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)