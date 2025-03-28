import torch
import torchvision
from dataset import get_data_loaders
from tqdm import tqdm
import argparse

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description="League of legends Doodle Classifier")

parser.add_argument('--model', choices=['mobilenet', 'resnet'], help="Specify the model type: 'mobilenet' or 'resnet'", required=True)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f'Using device {device}')

train_dataloader, val_dataloader = get_data_loaders(16, 256)

num_classes = 170

args = parser.parse_args()

if args.model == 'mobilenet':
    # Load MobileNet-V3-Small Model
    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(576, num_classes)
elif args.model == 'vgg':
    # Load VGG-16 Model
    model = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
elif args.model == 'cnn':
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, 1, 1),
        torch.nn.ReLU(True),
        torch.nn.MaxPool2d(2, 2),

        torch.nn.Conv2d(16, 32, 3, 1, 1),
        torch.nn.ReLU(True),
        torch.nn.MaxPool2d(2, 2),

        torch.nn.Conv2d(32, 64, 3, 1, 1),
        torch.nn.ReLU(True),
        torch.nn.MaxPool2d(2, 2),

        torch.nn.Conv2d(64, 128, 3, 1, 1),
        torch.nn.ReLU(True),
        torch.nn.MaxPool2d(2, 2),

        torch.nn.Flatten(),
        torch.nn.Linear(128*4*4, num_classes)
    )
    
else:
    # Load ResNet-50 Model
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(2048, num_classes)

model = model.to(device)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

if __name__ == '__main__':
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses = []
    test_accuracy = []
    test_losses = []
    epochs = 50
    lowest_loss = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}")
        train_l = train(train_dataloader, model, loss_fn, optimizer)
        test_l, test_a = test(val_dataloader, model, loss_fn)
        if test_l < lowest_loss:
            lowest_loss = test_l
            print('New lowest loss found: ', lowest_loss)
            model = model.to('cpu')
            torch.save(model, 'model.pth')
            model = model.to(device)

        train_losses.append(train_l)
        test_accuracy.append(test_a)
        test_losses.append(test_l)
    print("Done!")
