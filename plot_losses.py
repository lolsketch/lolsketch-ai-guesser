import matplotlib.pyplot as plt
import json
import numpy as np

with open('cnn_train_losses.json', 'r') as f:
    cnn_losses = json.load(f)

with open('resnet_train_losses.json', 'r') as f:
    resnet_losses = json.load(f)

with open('mobilenet_train_losses.json', 'r') as f:
    mobilenet_losses = json.load(f)

with open('vgg_train_losses.json', 'r') as f:
    vgg_losses = json.load(f)

def smooth_curve(values, weight=0.95):
    smoothed_values = []
    last = values[0]  # Initialize with the first value
    for val in values:
        last = weight * last + (1 - weight) * val  # Exponential moving average
        smoothed_values.append(last)
    return smoothed_values

cnn_losses_smooth = smooth_curve(cnn_losses)
resnet_losses_smooth = smooth_curve(resnet_losses)
mobilenet_losses_smooth = smooth_curve(mobilenet_losses)
vgg_losses_smooth = smooth_curve(vgg_losses)

plt.figure(figsize=(12, 6))
plt.xlabel('# of Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Graph of Convolutional Networks', fontsize=16)

plt.plot(cnn_losses_smooth, label='CNN')
plt.plot(resnet_losses_smooth, label='ResNet')
plt.plot(mobilenet_losses_smooth, label='MobileNet')
plt.plot(vgg_losses_smooth, label='VGG')

iterations = np.arange(len(cnn_losses))
plt.xlim(iterations[0], iterations[-1])

plt.legend(fontsize=10)
plt.show()