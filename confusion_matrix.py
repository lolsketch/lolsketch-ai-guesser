import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models import get_model
import argparse
from dataset import get_data_loaders
from tqdm import tqdm

def plot_confusion_matrix(model, dataloader, device, top_n=5):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Find most confused classes
    confused_sums = cm.sum(axis=1) - np.diag(cm)
    most_confused_indices = np.argsort(confused_sums)[-top_n:]

    # Extract top-N confusion matrix
    cm_top_n = cm[np.ix_(most_confused_indices, most_confused_indices)]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_top_n, annot=True, fmt="d", cmap="coolwarm", xticklabels=most_confused_indices, yticklabels=most_confused_indices)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Top-{top_n} ResNet Confusion Matrix")
    plt.show()


parser = argparse.ArgumentParser(description="League of legends Doodle Classifier")

parser.add_argument('--model', choices=['mobilenet', 'resnet', 'vgg', 'cnn'], help="Specify the model type: 'mobilenet', 'resnet', 'vgg', or 'cnn'", required=True)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model = get_model(model_type=args.model, pretrained=False)
model.load_state_dict(torch.load(f'models/{args.model}.pth', weights_only=True, map_location='cpu'))
model = model.to(device)
model.eval()

_, dataloader = get_data_loaders(256, 256)

plot_confusion_matrix(model, dataloader, device)
