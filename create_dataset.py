from draw_image import rasterize
import os
import json
from tqdm import tqdm

try:
    os.mkdir('dataset')
except:
    pass

with open('label_to_champion.json', 'r', encoding='utf-8') as file:
    label_to_champion = json.load(file)

idx = 0
file_names = os.listdir('image_data')
for f_name in tqdm(file_names):
    with open('image_data/' + f_name, 'r') as f:
        info = json.load(f)
        label = info['label']
        if label in label_to_champion:
            class_name = label_to_champion[label]
            try:
                os.mkdir(f'dataset/{class_name}')
            except:
                pass
            rasterize(info, f'dataset/{class_name}/{idx}.png')
            idx += 1
