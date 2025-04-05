from models import get_model
from torchvision.io.image import read_image
import os
import numpy as np
from sklearn.cluster import KMeans
import torch
from dataset import test_transform
import matplotlib.pyplot as plt
import torchvision

vgg = get_model(model_type='vgg', pretrained=True).features
vgg.eval()

device = 'mps'

vgg = vgg.to(device)

class_name = input('Enter Class Name: ')
num_centroids = int(input('Enter Number of Centroids: '))

file_names = os.listdir(f'dataset/train/{class_name}')

print(f'Plotting {len(file_names)} files from class {class_name}')

embeddings = np.empty(shape=(len(file_names), 2048))

with torch.no_grad():
    for idx, file_name in enumerate(file_names):
        image = read_image(f'dataset/train/{class_name}/{file_name}')
        image = test_transform(image).unsqueeze(0)
        image = image.to(device)
        embeddings[idx] = vgg(image)[0].reshape(-1).cpu().numpy()

kmeans = KMeans(n_clusters=num_centroids, init='k-means++')
kmeans.fit(embeddings)

centroids = kmeans.cluster_centers_

def get_closest_images(centroid):
    entries = []

    with torch.no_grad():
        for file_name in file_names:
            path = f'dataset/train/{class_name}/{file_name}'
            image = read_image(f'dataset/train/{class_name}/{file_name}')
            image = test_transform(image).unsqueeze(0)
            image = image.to(device)
            embedding = vgg(image)[0].reshape(-1).cpu().numpy()

            distance = np.linalg.norm(embedding - centroid)

            entries.append({
                'filepath': path,
                'distance': distance
            })
    
    entries.sort(key=lambda x: x['distance'])

    return entries[:4]

ticks = np.linspace(0, 64, 6)

plt.figure(figsize=(8, 8))
for centroid_idx in range(num_centroids):
    closest = get_closest_images(centroids[centroid_idx])
    for img_idx in range(len(closest)):
        image_path = closest[img_idx]['filepath']
        image = read_image(image_path)
        image = torchvision.transforms.ToPILImage()(image)
        plt.subplot(len(closest), num_centroids, img_idx * num_centroids + centroid_idx + 1)
        plt.imshow(image)
        plt.xticks([0, 16, 32, 48, 64], fontsize=8)
        plt.yticks([0, 16, 32, 48, 64], fontsize=8)
plt.show()
