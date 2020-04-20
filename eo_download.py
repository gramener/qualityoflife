import ee
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from tqdm import tqdm
import base64

from sentinel import mask2clouds, get_sentinel_image, ndvi, ndwi, ndbi, bare, nd_mask
from utils import read_df, read_images, get_image, save_image

ee.Initialize()
cwd = Path.cwd()
data_dir = cwd/'data'/'qol'
image_dir = data_dir/'sentinel'

# TODO
# 1. Read cluster file
# 2. Group by state and district
# 3. Loop through the region, get the image and save as thumbnail
# 4. Speed up the process

Coordinate = namedtuple('Coordinate', ['lat', 'lng'])

def process(row):
    coord, cluster = row
    area = 4000
    geometry = ee.Geometry.Point(coord.lng, coord.lat).buffer(area).bounds()
    roi = get_sentinel_image(geometry)
    roi=roi.visualize(**{'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'forceRgbOutput': True})
    
    
    url = roi.getThumbURL(params={'dimensions':"512x512", 'format':'jpg'})
    
    
    save_image(get_image(url), image_dir/f'{cluster}.png')

    # Compute the ND masks
    # ndvi_mask = nd_mask(ndvi(roi), 0.2)
    # ndwi_mask = nd_mask(ndwi(roi), 0.2)
    # ndbi_mask = nd_mask(ndbi(roi), 0.1)
    # bare_mask = nd_mask(bare(ndvi(roi), ndwi(roi)))

    # # Stack the RGB image and ND masks
    # stack = ee.ImageCollection([
    #     roi.visualize(**{'bands': ['B4', 'B3', 'B2'], 'min': 0.0, 'max': 0.3, 'forceRgbOutput': True}),
    #     ndwi_mask.visualize(**{'min': 0.5, 'max': 1, 'palette': ['cf92c6', '67001f'], 'forceRgbOutput': True}),
    #     ndvi_mask.visualize(**{'min': 0.5, 'max': 1, 'palette': ['98d7c7', '00441b'], 'forceRgbOutput': True}),
    #     ndbi_mask.visualize(**{'min': 0.5, 'max': 1, 'palette': ['fec253', '662506'], 'forceRgbOutput': True}),
    #     bare_mask.visualize(**{'min': 0, 'max': 1, 'forceRgbOutput': True})
    # ])

    # try:
    #     stack_url = ee.data.makeThumbUrl(ee.data.getThumbId({'image': stack.serialize(), 'dimensions': "512x512", 'format': 'png'}))
    #     save_image(get_image(stack_url), image_dir/f'{cluster}.png')
    # except ee.ee_exception.EEException as e:
    #     print(e)

def main():
    data = read_df(data_dir/'sentinel_main.csv')
    image_list = read_images(image_dir)

    for ((state, region), group) in data.groupby(['state', 'region']):
        with ThreadPoolExecutor(max_workers=5) as executor:
            rows = [(Coordinate(row['lat'], row['lng']), row['cluster'])
                    for _, row in group.iterrows() if str(row['cluster']) not in image_list]

            results = list(tqdm(executor.map(process, rows), desc=f'State: {state}, Region: {region}', total=len(rows)))
#        print(results)

        # for _, row in tqdm(group.iterrows(), desc=f'State: {state}, Region: {region}', total=group.shape[0]):
        #     coord, cluster, uor = Coordinate(row['lat'], row['lng']), row['cluster'], row['uor']
        #     if str(cluster) not in image_list:
        #         image_list.add(cluster)
        #         process(coord, cluster, uor)
        #     else:
        #         print(f'{cluster} present')


if __name__ == '__main__':
    main()



