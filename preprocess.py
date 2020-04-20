from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
from tqdm import tqdm


image_dir= Path('data/qol/sentinel')
output_dir = Path('data/qol/train')
output_dir.mkdir(exist_ok=True)

def split_image(image):
    d = {}
    for idx, label in zip(range(5), ('rgb', 'ndwi', 'ndvi', 'ndbi', 'bare')):
        d[label] = image.crop(box=(0, 512*idx, 512, 512*(idx+1)))

    return (d['rgb'], d['ndwi'], d['ndvi'], d['ndbi'], d['bare'])

def merge_layers(ndvi, ndwi, ndbi, bare):
    ndvi.paste(ndbi, (0, 0), ndbi)
    ndvi.paste(ndwi, (0, 0), ndwi)
    ndvi.paste(bare, (0, 0), bare)

    return ndvi

def process(path):
    image = Image.open(path)
    rgb, ndwi, ndvi, ndbi, bare = split_image(image)
    mask = merge_layers(ndvi, ndwi, ndbi, bare)
    rgb.save(output_dir/f'{path.stem}.png')
    mask.save(output_dir/f'{path.stem}-mask.png')

def main():
    image_list = list(image_dir.glob('260185.png'))

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(process, image_list), total=len(image_list)))

if __name__ == '__main__':
    main()

