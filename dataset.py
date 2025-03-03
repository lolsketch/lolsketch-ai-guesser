import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_data_loaders(train_bs, val_bs):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0.1),
        torchvision.transforms.RandomRotation(degrees=45),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.05),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.9568, 0.9540, 0.9552], std=[0.1607, 0.1632, 0.1631])
    ])

    dataset = torchvision.datasets.ImageFolder(root='dataset/train', transform=transform)

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    class_counts = np.zeros(len(dataset.classes)) 
    for _, label in train_dataset:
        class_counts[label] += 1

    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for _, label in train_dataset]

    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, sampler=sampler)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_bs, shuffle=False)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    dataset = torchvision.datasets.ImageFolder(root='dataset/test', transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    classes = list(dataset.class_to_idx.keys())

    images, labels = next(iter(dataloader))

    plt.figure(figsize=(8, 8))
    plt.title('Testing Data')
    plt.axis('off')
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(torchvision.transforms.ToPILImage()(images[i]))
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(classes[labels[i]])
    plt.show()