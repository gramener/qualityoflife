import pandas as pd
import re
import requests
from typing import Iterable

def read_df(filename):
    return pd.read_csv(filename)

def read_images(directory):
    return {path.stem for path in directory.glob('*.png')}

def get_image(url):
    return requests.get(url).content

def save_image(img, filename):
    '''Saves image in the given filename.'''
    with open(filename, 'wb') as f:
        f.write(img)

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')

def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()
