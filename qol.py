import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image

from output import single_predict, w2s, d2s, p2s, t2s
from eo_download_one import process
# import time
# def process(lat, lng):
#     time.sleep(3)
#     return True

PATH = Path('data/qol')
DATA_PATH = PATH/'sentinel.csv'
IMGS_PATH = PATH/'sentinel'
TEST_IMAGES = Path('test_images')

m2m = {
    'Wealth': {'metric': 'wealth', 'mapper': w2s, 'phrase': 'Median Wealth Levels for Clusters'},
    'Drought': {'metric': 'drought', 'mapper': d2s, 'phrase': 'Drought Index for Clusters'},
    'Population Density': {'metric': 'pop_density', 'mapper': p2s, 'phrase': 'Population Distribution for Clusters'},
    'Toilet Facility': {'metric': 'toilet_type', 'mapper': t2s, 'phrase': 'Most common toilet types for Clusters'},
}

@st.cache
def load_data(nrows=100):
    data = pd.read_csv(DATA_PATH, nrows=nrows)
    data.rename(columns={'lng': 'lon'}, inplace=True)
    return data

def main():
    st.title('Quality of Life from Satellite Imagery')

    st.sidebar.title('Configuration')
    show_data = st.sidebar.checkbox('Show Ground Truth data', value=False)
    show_config = st.sidebar.selectbox('Data Source', ['Select from list', 'Enter LAT,LON coordinate'])

    if show_config == 'Select from list':
        PLACE = st.sidebar.selectbox('Select a place from the dropdown menu', [name.stem for name in TEST_IMAGES.iterdir() if name.suffix == '.jpeg'])
    else:
        COORDS = st.sidebar.text_input('Enter lat,lng coords of a region', '12.9542946,77.4908552')

    METRIC = st.sidebar.radio('Select the metric you would like to visualize',
                             ('Wealth', 'Drought', 'Population Density', 'Toilet Facility'),
                               index=0)

    # DATA
    if show_data:
        state = st.text('Loading data...')
        data = load_data(100)
        state.text('Loading data... done!')
        st.subheader('DHS VII Survey Dataset, 2015 - 2016')
        st.write('Cluster level aggregated data based on filtered metrics')
        st.dataframe(data)

    #[TODO] Show Few Satellite Images


    # MODEL
    if show_config == 'Select from list':
        st.markdown(f"You are looking at **{m2m[METRIC]['phrase']}** in **{PLACE}**")
        filename = TEST_IMAGES/f'{PLACE}.jpeg'
    else:
        try:
            LAT,LNG = map(float, COORDS.split(','))
            st.write(LAT, LNG)
            # state = st.text(f'Downloading Sentinel 2 MSI Imagery for Location({LAT},{LNG})')
            state = process(LAT,LNG)
            st.write(state)
            # state.text('Fetching... Done!')
            # st.markdown(f"You are looking at **{m2m[METRIC]['phrase']}** at **LOCATION({LAT},{LNG})**")
            filename = 'temp.jpeg'
        except Exception as e:
            st.error(e)
            # st.error('Please enter the coordinates in (LAT,LNG) format')


    state = st.text('Baking the recipe for the model...')
    source, pred = single_predict(filename, 4, **m2m[METRIC])
    state.text('Recipe... Ready!')

    st.write(f'Sentinel 2-MSI Imagery.')
    st.image(source, use_column_width=True)

    st.write(f"Overlayed layer for {m2m[METRIC]['phrase']}")
    st.image('./Wealth Index@3x.png', use_column_width=True)
    st.image(pred, use_column_width=True)


    if st.button('Done! Thank You!'):
        st.balloons()



if __name__ == "__main__":
    main()
