import ee

def mask2clouds(image):
    '''Mask the clouds from Sentinel 2 Imagery'''
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0) and (qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)

def get_sentinel_image(geometry, start_date='2015-06-01', end_date='2017-12-31', top=10):
    return ee.ImageCollection('COPERNICUS/S2')\
            .filterDate(start_date, end_date)\
            .filterBounds(geometry)\
            .sort('CLOUDY_PIXEL_PERCENTAGE').limit(top)\
            .map(mask2clouds)\
            .select(['B4', 'B3', 'B2', 'B8', 'B11', 'B12'])\
            .median()\
            .clip(geometry)


def ndvi(sat_img):
    return sat_img.normalizedDifference(['B8', 'B4']) # NDVI: RED - NIR

def ndwi(sat_img):
    return sat_img.normalizedDifference(['B3', 'B8']) # NDVI: NIR - GREEN

def ndbi(sat_img):
    return sat_img.normalizedDifference(['B11', 'B8']) # NDBI: SWIR1 - NIR

def bare(ndvi, ndwi):
    return ndvi.lt(0.2).And(ndwi.lt(0))

def nd_mask(nd, threshold=None):
    if threshold:
        return nd.updateMask(nd.gte(threshold))
    else:
        return nd.updateMask(nd)
