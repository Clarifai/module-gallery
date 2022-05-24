import streamlit as st
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from stqdm import stqdm

from utils.mosaic import urls_to_mosaic


##########################################################
def display():
  # This must be within the display() function.
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = auth.get_stub()
  metadata = auth.metadata
  userDataObject = auth.get_user_app_id_proto()
  lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size=16)

  st.title("Image Mosaic Builder")
  with st.form(key="mosiac-inputs"):

    mtotal = st.number_input(
        "Select number of images to add to mosaic:", min_value=1, max_value=100)
    submitted = st.form_submit_button('Submit')

  if submitted:
    if mtotal is None or mtotal == 0:
      st.warning("Number of images must be provided.")
      st.stop()
    else:
      st.write("Mosaic number of images will be: {}".format(mtotal))

    total = st.session_state['total']

    # Stream inputs from the app
    all_images = []
    for inp in stqdm(
        lister.inputs_generator(), desc="Listing all the inputs in the app", total=total):
      if inp.data.image is not None:
        all_images.append(inp.data.image)
      if len(all_images) >= mtotal:
        break

    print(all_images)
    url_list = [im.url for im in all_images if im.url != ""]

    mosaic = urls_to_mosaic(url_list)

    st.image(mosaic)
