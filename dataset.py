import torchvision
import torch
import matplotlib.pyplot as plt
import random
import os
from torchvision.io.image import read_image

# Pre-computed mean and std from running calculate_mean_and_std.py
MEAN = [0.9568, 0.9540, 0.9552]
STD = [0.1607, 0.1632, 0.1631]

# This dataset class allows us to sample from our dataset using weighted random sampling.
# This helps us deal with the imbalanced problem we have making all classes have the same probability of being selected.
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.transform = transform
        self.split = split
        self.root = root
        self.class_names = [
                f for f in os.listdir(os.path.join(root, split)) 
                if os.path.isdir(os.path.join(root, split, f))
            ]
        self.class_file_names = {}

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        self.num_samples = 0
        for class_name in self.class_names:
            file_names = [
                f for f in os.listdir(os.path.join(root, split, class_name)) 
                if f.endswith(".png")
            ]
            self.class_file_names[class_name] = file_names
            self.num_samples += len(file_names)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        class_name = random.choice(self.class_names)
        file_name = random.choice(self.class_file_names[class_name])
        image = read_image(os.path.join(self.root, self.split, class_name, file_name))
        image = self.transform(image), self.class_to_idx[class_name]
        return image

def get_data_loaders(train_bs, val_bs):
    # Apply some transformations to prevent overfitting and improve generalization
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomCrop((64, 64)),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0.1),
        torchvision.transforms.RandomRotation(degrees=45),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.05),
        torchvision.transforms.RandomGrayscale(p=0.2), # This could be increased to .5
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MEAN, STD)
    ])

    # We dont need any data augmentation for testing
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((64, 64)), # Resize images to 64x64 instead of cropping
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MEAN, STD)
    ])

    # Right now we are using the same 
    train_dataset = Dataset('dataset', 'train', transform=train_transform)
    val_dataset = Dataset('dataset', 'test', transform=test_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=False, num_workers=2) # No need to shuffle since we are ignoring the index in __getitem__
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_bs, shuffle=False, num_workers=2)

    return train_dataloader, val_dataloader

def denorm(images):
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return images * std + mean

if __name__ == '__main__':
    _, test_dl = get_data_loaders(16, 16)

    classes = list(test_dl.dataset.class_names)

    images, labels = next(iter(test_dl))

    images = denorm(images) # Denormalize the images to view their original colors

    # Plot the images along with their labels
    fig, axes = plt.subplots(1, 4, figsize=(8, 3))  # 1 row, 4 columns
    plt.suptitle('Testing Data')

    # Iterate through images and labels, and plot them
    for i in range(4):
        ax = axes[i]  # Get the current axis
        ax.imshow(torchvision.transforms.ToPILImage()(images[i]))
        ax.set_xlabel(classes[labels[i]], fontsize=10)  # Set the label with a smaller font size
        ax.set_xticks(range(0, images[i].shape[2], 10))  # Increase x-tick frequency
        ax.set_yticks(range(0, images[i].shape[1], 10))  # Increase y-tick frequency
        ax.tick_params(axis='both', labelsize=8)  # Reduce font size of ticks

    # Show the plot
    plt.tight_layout()
    plt.show()