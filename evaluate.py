import torch
import argparse
from dataset import get_data_loaders
from sklearn.metrics import top_k_accuracy_score
from models import get_model

parser = argparse.ArgumentParser(description="Top-N Accuracy Estimation")

parser.add_argument('--model', choices=['mobilenet', 'resnet', 'vgg', 'cnn'], help="Specify the model type: 'mobilenet', 'resnet', 'vgg', or 'cnn'", required=True)

args = parser.parse_args()

# Load the data
_, dataloader = get_data_loaders(512, 256)

model = get_model(model_type=args.model, pretrained=False)
model.load_state_dict(torch.load(f'models/{args.model}.pth', weights_only=True, map_location='cpu'))
model.eval()

all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in dataloader:
        preds = model(images)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(preds.cpu().numpy())

top_1 = top_k_accuracy_score(all_labels, all_probs, k=1)
top_5 = top_k_accuracy_score(all_labels, all_probs, k=5)
top_10 = top_k_accuracy_score(all_labels, all_probs, k=10)

print(f'Top-1: {round(top_1 * 100, 1)}')
print(f'Top-5: {round(top_5 * 100, 1)}')
print(f'Top-10: {round(top_10 * 100, 1)}')
