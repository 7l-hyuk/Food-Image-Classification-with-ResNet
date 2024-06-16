import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.conv4 = nn.Conv2d(20, 25, 5)
        self.conv5 = nn.Conv2d(25, 30, 5)

        self.fc1 = nn.Linear(30*41*41, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepNet10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.conv4 = nn.Conv2d(20, 25, 5)
        self.conv5 = nn.Conv2d(25, 30, 5)
        self.conv6 = nn.Conv2d(30, 32, 3)
        self.conv7 = nn.Conv2d(32, 34, 3)
        self.conv8 = nn.Conv2d(34, 38, 2)
        self.conv9 = nn.Conv2d(38, 42, 2)
        self.conv10 = nn.Conv2d(42, 46, 2)

        self.fc1 = nn.Linear(46*34*34, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SkipConDeep10(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 13, 5)
        self.conv3 = nn.Conv2d(13, 18, 5)
        self.conv4 = nn.Conv2d(18, 23, 5)
        self.conv5 = nn.Conv2d(23, 26, 5)
        self.conv6 = nn.Conv2d(26, 33, 5)
        self.conv7 = nn.Conv2d(33, 37, 5)
        self.conv8 = nn.Conv2d(37, 43, 5)
        self.conv9 = nn.Conv2d(43, 47, 5)
        self.conv10 = nn.Conv2d(47, 53, 5)

        self.fc1 = nn.Linear(53*13*13, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        p1d = (0, 0, 0, 0, 5, 5)
        shortcut = F.pad(shortcut, p1d, "constant", 0)
        m = nn.ZeroPad2d(4)
        shortcut += m(x)

        x = F.relu(self.conv3(shortcut))
        x = F.relu(self.conv4(x))

        shortcut = F.pad(shortcut, p1d, "constant", 0)
        shortcut += m(x)

        x = F.relu(self.conv5(shortcut))
        x = F.relu(self.conv6(x))

        shortcut = F.pad(shortcut, p1d, "constant", 0)
        shortcut += m(x)

        x = F.relu(self.conv7(shortcut))
        x = F.relu(self.conv8(x))

        shortcut = F.pad(shortcut, p1d, "constant", 0)
        shortcut += m(x)

        x = self.pool(F.relu(self.conv9(shortcut)))
        x = self.pool(F.relu(self.conv10(x)))
        x = self.pool(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1
            ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
            )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                    ),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: BasicBlock,
            num_blocks: list,
            num_classes: int = 10
            ):

        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*7*7, num_classes)

    def _make_layer(
            self,
            block: BasicBlock,
            planes: int,
            num_blocks: int,
            stride: int
            ) -> nn.Sequential:

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
