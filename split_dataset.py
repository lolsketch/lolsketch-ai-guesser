import os
import shutil

class_names = os.listdir('dataset')

os.mkdir('dataset/train')
os.mkdir('dataset/test')

for cn in class_names:
    file_names = os.listdir(f'dataset/{cn}')
    num_train = int(len(file_names) * 0.95)
    os.mkdir(f'dataset/train/{cn}')
    os.mkdir(f'dataset/test/{cn}')
    for fn in file_names[:num_train]:
        shutil.move(f'dataset/{cn}/{fn}', f'dataset/train/{cn}/{fn}')
    for fn in file_names[num_train:]:
        shutil.move(f'dataset/{cn}/{fn}', f'dataset/test/{cn}/{fn}')
    os.rmdir(f'dataset/{cn}')
