import torch
import numpy as np
import os
from torchvision.io.image import read_image
from dataset import test_transform
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse
from models import get_model
from constants import *

means = 5
neighbors = 50
class_names = os.listdir('dataset/train')

parser = argparse.ArgumentParser(description="KNN++ with Feature Embeddings")

parser.add_argument('--model', choices=['mobilenet', 'resnet', 'vgg'], help="Specify the model type: 'mobilenet', 'resnet', or 'vgg'", required=True)

args = parser.parse_args()

model = get_model(model_type=args.model, pretrained=True)

if args.model == 'mobilenet':
    model = torch.nn.Sequential(*list(model.children())[:-1])
    embedding_dim = 576
elif args.model == 'vgg':
    model = model.features
    embedding_dim = 2048
else:
    model = torch.nn.Sequential(*list(model.children())[:-1])
    embedding_dim = 2048

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model = model.to(device)
model.eval()

def get_centroids_for_label(data_dir, label):
    filenames = os.listdir(os.path.join(data_dir, label))

    datapoints = np.empty(shape=(len(filenames), embedding_dim), dtype=np.float32)

    batch = []
    batch_size = 64

    with torch.no_grad():
        for i, fn in enumerate(filenames, 1):
            image = read_image(os.path.join(data_dir, label, fn))
            image = test_transform(image)
            image = image.unsqueeze(0)
            batch.append(image)
            
            if i % batch_size == 0 or i == len(filenames):
                batch_tensor = torch.cat(batch, dim=0)
                batch_tensor = batch_tensor.to(device)
                embedding = model(batch_tensor).reshape(len(batch), -1)
                datapoints[i-len(batch):i] = embedding.cpu().numpy()
                batch = []

    # run kmeans clustering now
    kmeans = KMeans(n_clusters=means, init='k-means++')
    kmeans.fit(datapoints)

    return kmeans.cluster_centers_

def get_all_centroids(data_dir, split):
    centroids = np.empty(shape=(num_classes, means, embedding_dim), dtype=np.float32)

    for i, cn in enumerate(class_names, 0):
        centroids[i] = get_centroids_for_label(os.path.join(data_dir, split), cn)

    return centroids

def get_all_embeddings(data_dir, split):
    class_names = os.listdir(os.path.join(data_dir, split))

    data = []
    labels = []

    with torch.no_grad():
        for cn in class_names:
            if cn == '.DS_Store': # Skip empty macos files
                continue
            filenames = os.listdir(os.path.join(data_dir, split, cn))
            for fn in filenames:
                path = os.path.join(data_dir, split, cn, fn)
                image = read_image(path)
                image = test_transform(image)
                image = image.unsqueeze(0)
                image = image.to(device)
                embedding = model(image).reshape(1, embedding_dim).cpu().numpy()
                data.append(embedding)
                labels.append(class_names.index(cn))

    data = np.concatenate(data, axis=0)
    labels = np.array(labels)

    return data, labels

def rank_weighting(distances):
    ranks = np.argsort(distances)
    weights = 1 / np.sqrt(np.arange(1, len(distances) + 1))
    return weights[ranks]

if __name__ == '__main__':
    train_centroids = get_all_centroids('dataset', 'train')
    x_test, y_test = get_all_embeddings('dataset', 'test')

    # # Reshape X_train to (510, 2048)
    x_train = train_centroids.reshape(-1, embedding_dim)

    # # Generate class labels (repeat each class label 5 times)
    y_train = np.repeat(np.arange(num_classes), means)

    knn = KNeighborsClassifier(n_neighbors=neighbors, weights=rank_weighting)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100}%")
