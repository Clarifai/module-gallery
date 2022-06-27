from itertools import cycle

import numpy as np
import streamlit as st
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from clarifai_utils.modules.css import ClarifaiStreamlitCSS
from stqdm import stqdm

from utils.mosaic import download_urls


##########################################################
def display():
  ClarifaiStreamlitCSS.insert_default_css(st)

  # This must be within the display() function.
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = auth.get_stub()
  metadata = auth.metadata
  userDataObject = auth.get_user_app_id_proto()
  lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size=16)
  st.title("Grid Input Viewer")
  with st.form(key="data-inputs"):
    mtotal = st.number_input(
        "Select number of images to view as a grid:", min_value=1, max_value=100)
    submitted = st.form_submit_button('Submit')

  if submitted:
    if mtotal is None or mtotal == 0:
      st.warning("Number of images must be provided.")
      st.stop()
    else:
      st.write("Number of images in grid will be: {}".format(mtotal))

    # Stream inputs from the app
    all_images = []
    for inp in stqdm(
        lister.inputs_generator(), desc="Listing all the inputs in the app", total=mtotal):
      if inp.data.image is not None:
        all_images.append(inp.data.image)
      if len(all_images) >= mtotal:
        break

    url_list = [im.url for im in all_images if im.url != ""]

    filteredImages = [tup[1] for tup in download_urls(url_list)]

    print(filteredImages)
    # caption = [] # your caption here
    cols = cycle(
        st.columns(4))  # st.columns here since it is out of beta at the time I'm writing this
    for idx, filteredImage in enumerate(filteredImages):
      next(cols).image(np.array(filteredImage), use_column_width=True)


if __name__ == '__main__':
  display()
