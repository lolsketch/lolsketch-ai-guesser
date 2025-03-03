import base64
import json
import os
from PIL import Image, ImageDraw

def rasterize(info, out_path):
  data = list(base64.b64decode(info['data']))

  img = Image.new('RGB', (256, 256), 'white')
  draw = ImageDraw.Draw(img)

  idx = 0
  while True:
    if idx >= len(data):
      break
    if data[idx] == 0:
      x = data[idx+1]
      y = data[idx+2]
      x1 = data[idx+3]
      y1 = data[idx+4]
      width = int(data[idx+5] / 4.25)
      color = (data[idx+6], data[idx+7], data[idx+8])

      radius = width / 2
      draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
      draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=color)
      draw.line([(x, y), (x1, y1)], color, width)

      idx += 9
    elif data[idx] == 1:
      x = data[idx+1]
      y = data[idx+2]
      color = (data[idx+3], data[idx+4], data[idx+5])
      ImageDraw.floodfill(img, (x, y), color)
      idx += 6

  img.resize((80, 80)).save(out_path)

if __name__ == '__main_':
  for name in os.listdir('image_data'):
    if name[-4:] == 'json':
      json_path = 'image_data/' + name
      out_path = 'rasterized_images/' + name[:-5] + '.png'
      try:
        rasterize(json_path, out_path)
      except:
        print('Failed at: ' + json_path)
