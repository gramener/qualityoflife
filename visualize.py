import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw
from random import randint

image_dir= Path('test_images/')
output_dir = Path('test_images/')
output_dir.mkdir(exist_ok=True)

def split(img, ntiles):
    np_img = np.array(img)
    h,w,c = np_img.shape
    s = h // ntiles
    d = defaultdict(dict)

    for i in range(ntiles):
        for j in range(ntiles):
            d[str(i)+str(j)]['img'] = Image.fromarray(np_img[s*i:s*(i+1), s*j:s*(j+1),:])

    return d

def merge(d, ntiles, metric, m2s):
    s = d['00']['img'].size[0]

    img = Image.new('RGBA', (s*ntiles,s*ntiles))

    for i in range(ntiles):
        for j in range(ntiles):
            base = d[str(i)+str(j)]['img'].convert('RGBA')
            score = d[str(i)+str(j)]['score'][metric]

            rect = Image.new('RGBA', base.size, (255,255,255,255))
            r = ImageDraw.Draw(rect)
            r.rectangle([(0,0), base.size], fill=m2s[score], outline=(0,0,0,128), width=4)
            out = Image.alpha_composite(base, rect)

            img.paste(out, box=(s*j, s*i))

    return img

def merge_final(d, ntiles, m2s):
    s = d['00']['img'].size[0]

    img = Image.new('RGBA', (s*ntiles,s*ntiles))

    for i in range(ntiles):
        for j in range(ntiles):
            base = d[str(i)+str(j)]['img'].convert('RGBA')
            score = d[str(i)+str(j)]['score']

            rect = Image.new('RGBA', base.size, (255,255,255,255))
            r = ImageDraw.Draw(rect)
            r.rectangle([(0,0), base.size], fill=m2s[score], outline=(0,0,0,128), width=4)
            out = Image.alpha_composite(base, rect)

            img.paste(out, box=(s*j, s*i))

    return img

def main():
    img = Image.open(image_dir/'amravati-16km.jpeg')
    tiles = split(img, 4)
    new_img = merge(tiles, 4)

    new_img.save('amravati-sample.png')

if __name__ == '__main__':
    main()
