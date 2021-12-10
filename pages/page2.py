import streamlit as st
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
## Import in the Clarifai gRPC based objects needed
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from stqdm import stqdm

from utils.mosaic import urls_to_mosaic

########################
# Required in every Clarifai streamlit app
########################
# Validate and parse the query params we need.
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
