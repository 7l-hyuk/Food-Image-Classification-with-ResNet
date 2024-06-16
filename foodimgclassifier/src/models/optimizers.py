import torch.optim as optim
import torch.nn as nn

from models.models import (
    ShallowNet,
    DeepNet5,
    DeepNet10,
    SkipConDeep10,
    ResNet18
    )

models = [
    shallownet := ShallowNet(),
    deepnet5 := DeepNet5(),
    deepnet10 := DeepNet10(),
    skipcondeep10 := SkipConDeep10(),
    resnet18 := ResNet18(),
]

criterion = nn.CrossEntropyLoss()


def make_optim(model: nn.Module) -> optim.SGD:
    return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


shallow_optimizer = make_optim(shallownet)
deep5_optimizer = make_optim(deepnet5)
deep10_optimizer = make_optim(deepnet10)
skipcondeep10_optimizer = make_optim(skipcondeep10)
resnet18_optimizer = make_optim(resnet18)
