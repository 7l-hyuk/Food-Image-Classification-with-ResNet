from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data import random_split

trans = transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

dataset = torchvision.datasets.ImageFolder(
    root="C:/Users/7lhyu/Documents/foodimgclassifier/data",
    transform=trans
)

classes = dataset.classes
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    )
test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=True,
    )

if __name__ == "__main__":
    print(f"CLASSES:\n{classes}")
    trainiter = iter(train_loader)
    train_images, train_labels = next(trainiter)
    print(f"shuffled labels from train_loader: {train_labels}")

    testiter = iter(test_loader)
    test_images, test_labels = next(testiter)
    print(f"shuffled labels from test_loader: {test_labels}")
