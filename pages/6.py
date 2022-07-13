from itertools import cycle

import pandas as pd
import numpy as np
import streamlit as st
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from clarifai_utils.modules.css import ClarifaiStreamlitCSS
from stqdm import stqdm

from utils.mosaic import download_urls


##########################################################
def display():
  ClarifaiStreamlitCSS.insert_css_file("style.css", st)

  st.header("Widgets")

  st.button('button')
  st.download_button('download button', 'download.csv')
  st.checkbox('checkbox')
  st.radio('radio', ['option 1', 'option 2'])
  st.selectbox('selectbox', ['option 1', 'option 2'])
  st.multiselect('multiselect', ['option 1', 'option 2'])
  st.slider('slider', 1.0, 5.0)
  st.select_slider('select slider', range(5))
  st.text_input('text input')
  st.number_input('number input')
  st.text_area('text area')
  st.date_input('date input')
  st.time_input('time input')
  st.file_uploader('file uploader', type='csv')


if __name__ == '__main__':
  display()
