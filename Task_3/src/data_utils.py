import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    raw_transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    raw_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=raw_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, trainset, raw_trainset, classes


def get_data_stats(raw_dataset, scaled_dataset):
    raw_img, _ = raw_dataset[0]
    scaled_img, _ = scaled_dataset[0]

    return {
        'raw_min': float(raw_img.min()),
        'raw_max': float(raw_img.max()),
        'scaled_min': float(scaled_img.min()),
        'scaled_max': float(scaled_img.max())
    }