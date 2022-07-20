import streamlit as st
## Import in the Clarifai gRPC based objects needed
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.modules.css import ClarifaiStreamlitCSS

########################
# Required in every Clarifai streamlit app
########################

st.set_page_config(layout="wide")

ClarifaiStreamlitCSS.insert_default_css(st)

# Validate and parse the query params we need.
auth = ClarifaiAuthHelper.from_streamlit(st)
########################

st.title("Welcome to the Clarifai Module Gallery")
st.write(
    "This landing page have any significant functioality as we encourage you to implement your applications as multi-page apps."
)
st.write("Take a look at the pages/*.py files.")
###################
