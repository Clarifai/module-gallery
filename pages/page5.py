from itertools import cycle

import numpy as np
import streamlit as st
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
## Import in the Clarifai gRPC based objects needed
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
## Import in the Clarifai gRPC based objects needed
from stqdm import stqdm

from utils.mosaic import download_urls

try:
  auth = ClarifaiAuthHelper.from_streamlit_query_params(st.experimental_get_query_params())
except:
  auth = ClarifaiAuthHelper.from_env()

# Create the clarifai grpc client.
channel = ClarifaiChannel.get_grpc_channel(base="api.clarifai.com")
stub = service_pb2_grpc.V2Stub(channel)
metadata = auth.metadata
print(metadata)
userDataObject = auth.get_user_app_id_proto()
print(userDataObject)

lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size=16)


##########################################################
def display():
  with st.form(key="data-inputs"):
    mtotal = st.number_input("Select number of images to view:", min_value=1, max_value=100)
    submitted = st.form_submit_button('Submit')

  if submitted:
    if mtotal is None or mtotal == 0:
      st.warning("Number of images must be provided.")
      st.stop()
    else:
      st.write("Mosaic number of images will be: {}".format(mtotal))

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
