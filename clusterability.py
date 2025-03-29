import torch
from sklearn.metrics import silhouette_score
import numpy as np
from dataset import get_data_loaders
import torchvision
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description="Silhouette Score Calculator")

parser.add_argument('--model', choices=['mobilenet', 'resnet', 'vgg', 'cnn'], help="Specify the model type: 'mobilenet' or 'resnet'", required=True)

args = parser.parse_args()

# Load the data
dataloader, _ = get_data_loaders(512, 64)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

num_classes = 170

if args.model == 'mobilenet':
    # Load MobileNet-V3-Small Model
    model = torchvision.models.mobilenet_v3_small()
    model.classifier = torch.nn.Linear(576, num_classes)
    model.load_state_dict(torch.load('models/mobilenet.pth', weights_only=True, map_location='cpu'))
    model = torch.nn.Sequential(*list(model.children())[:-1])

elif args.model == 'vgg':
    # Load VGG-16 Model
    model = torchvision.models.vgg16()
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    model.load_state_dict(torch.load('models/vgg.pth', weights_only=True, map_location='cpu'))
    model = model.features
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
    model.load_state_dict(torch.load('models/cnn.pth', weights_only=True, map_location='cpu'))
    model = model[:-2]

else:
    # Load ResNet-50 Model
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load('models/resnet.pth', weights_only=True, map_location='cpu'))
    model = torch.nn.Sequential(*list(model.children())[:-2])


model.eval()
model = model.to(device)

# Extract embeddings or predictions from the model
all_embeddings = []
all_labels = []

# Iterate over the test_dataloader to collect embeddings and labels
with torch.no_grad():
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.reshape(outputs.size(0), -1) # Flatten embedding feature vector
        all_embeddings.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())

# Convert the list of embeddings and labels into numpy arrays
all_embeddings = np.concatenate(all_embeddings, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Compute the silhouette score using true labels
score = silhouette_score(all_embeddings, all_labels)
print(f'Silhouette Score: {score}')



# Clusterability Visualization with t-SNE

unique_labels = np.unique(all_labels)
selected_labels = unique_labels[:5]

mask = np.isin(all_labels, selected_labels)

filtered_embeddings = all_embeddings[mask]
filtered_labels = all_labels[mask]

tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(filtered_embeddings)

plt.figure(figsize=(8, 6))
for label in selected_labels:
    idx = filtered_labels == label
    plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=f'Class {label}', alpha=0.7)

plt.title('t-SNE Visualization of Model Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
