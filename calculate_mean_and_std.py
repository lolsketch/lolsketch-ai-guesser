from dataset import get_data_loaders
import torch

def calculate_mean_std(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in dataloader:
        images = images.float()

        
        mean += images.mean(dim=[0, 2, 3])
        std += (images**2).mean(dim=[0, 2, 3])

        total_pixels += images.numel()

    mean /= len(dataloader)
    std = torch.sqrt(std / len(dataloader) - mean**2)

    return mean, std


if __name__ == '__main__':
    dataloader, _ = get_data_loaders(32, 1)
    mean, std = calculate_mean_std(dataloader)
    print('Mean:', mean)
    print('Std:', std)
