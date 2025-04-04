import torch
from dataset import get_data_loaders
from tqdm import tqdm
import argparse
from models import get_model
import json


parser = argparse.ArgumentParser(description="League of legends Doodle Classifier")

parser.add_argument('--model', choices=['mobilenet', 'resnet', 'vgg', 'cnn'], help="Specify the model type: 'mobilenet', 'resnet', 'vgg', or 'cnn'", required=True)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f'Using device {device}')

train_dataloader, val_dataloader = get_data_loaders(16, 256)

num_classes = 170

args = parser.parse_args()

if args.model == 'cnn':
    model = get_model(model_type=args.model, pretrained=False)
else:
    model = get_model(model_type=args.model, pretrained=True)

model = model.to(device)

# Define the train and test functions
def train(dataloader, model, loss_fn, optimizer):
    model.train()

    losses = []

    avg_loss = 0

    idx = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        idx += 1
        if idx == 10:
            idx = 0
            avg_loss /= 10
            losses.append(avg_loss)
            avg_loss = 0
            
    return losses
        

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
    return test_loss

# Define Loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Store train_losses and accuracy
train_losses = []
epochs = 35

# Keep track of the lowest loss to checkpoint the model
lowest_loss = 1000

for t in range(epochs):
    print(f"Epoch {t+1}")
    train_loss = train(train_dataloader, model, loss_fn, optimizer) 
    test_loss = test(val_dataloader, model, loss_fn)
    
    if test_loss < lowest_loss:
        lowest_loss = test_loss
        print('New lowest loss found: ', lowest_loss)

        # torch.save(model.state_dict(), f'models/{args.model}.pth')

    train_losses.extend(train_loss)

with open('train_losses.json', 'w') as f:
    json.dump(train_losses, f)

print("Done!")
