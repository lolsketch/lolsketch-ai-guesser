import torch
import torchvision
import matplotlib.pyplot as plt

def acc_given_n_guesses(model, dataloader, n, device):
    correct = 0
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)

            preds = model(image)[0]

            _, prediction_indices = torch.sort(preds, descending=True)

            for p_idx in prediction_indices[:n]:
                if p_idx == label:
                    correct += 1
                    break
            
    return correct / len(dataloader)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.load('model.pth')
    model = model.to(device)
    model.eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.9568, 0.9540, 0.9552], std=[0.1607, 0.1632, 0.1631])
    ])

    dataset = torchvision.datasets.ImageFolder(root='dataset/test', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    accuracies = []
    for i in range(5):
        accuracies.append(acc_given_n_guesses(model, dataloader, i+1, device))
        print('done:', i)
    
    plt.figure(figsize=(8, 8))
    plt.title('Accuracy Given N Guesses')
    plt.plot(accuracies)
    plt.xlabel('Number of Guesses')
    plt.ylabel('Accuracy')
    plt.show()

    print(accuracies)