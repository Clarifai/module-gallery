import glob
import importlib

import streamlit as st
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister

########################
# Required in every Clarifai streamlit app
########################
# Validate and parse the query params we need.
try:
  auth = ClarifaiAuthHelper.from_streamlit_query_params(st.experimental_get_query_params())
except:
  auth = ClarifaiAuthHelper.from_env()

st.set_page_config(layout="wide")
########################

# Create the clarifai grpc client.
channel = ClarifaiChannel.get_grpc_channel(base="api.clarifai.com")
stub = service_pb2_grpc.V2Stub(channel)
metadata = auth.metadata
print(metadata)
userDataObject = auth.get_user_app_id_proto()
print(userDataObject)

page_size = 3

lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size)

# current_page.display()

###################

# Shared API calls before anything goes

if 'get_input_count_response' not in st.session_state:
  get_input_count_response = stub.GetInputCount(
      service_pb2.GetInputCountRequest(user_app_id=userDataObject), metadata=metadata)
  if get_input_count_response.status.code != status_code_pb2.SUCCESS:
    raise Exception("Get input count failed, response: %r" % get_input_count_response)
  counts = get_input_count_response.counts
  print(counts)
  total = get_input_count_response.counts.processed + get_input_count_response.counts.errors + get_input_count_response.counts.errors + get_input_count_response.counts.to_process
  print(total)

  st.session_state['counts'] = counts
  st.session_state['total'] = total
  st.session_state['get_input_count_response'] = get_input_count_response


####################################
# Style the page to match Clarifai styles.
####################################
def local_css(file_name):
  with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")
####################################

####################################
# Get the current page to render
####################################
# Read the page param from the query bar.
qp = st.experimental_get_query_params()
# List all the available pages.
page_files = sorted(glob.glob("pages/page*.py"))
page_numbers = [f.replace('pages/page', '').replace('.py', '') for f in page_files]
N = len(page_files)
# Get the page or default to 1 from the url.
page = qp.get('page', ['1'])[0]
# Check that the page number coming in is within the range of pages in the folder.
if page not in page_numbers:
  raise Exception(
      "Page %s is out of range, no page in pages/ folder found with this name. Valid page numbers are: %s"
      % (page, str(page_numbers)))


######
# This section is optional, but you can render a dropdown to jump between pages in a multi-page
# app. This is handy when you're not rending the app mode in our UI with the sidebar.
######
def callback():
  # Callbacks use the session state and store they value in the "key" param of the given UI
  # element, in this case "page_selector".
  result = st.session_state.page_selector
  # update the query params.
  qp = st.experimental_get_query_params()
  qp['page'] = [result]
  st.experimental_set_query_params(**qp)


cols = st.columns(4)
result = cols[0].selectbox(
    'Select the page to jump to',
    page_numbers,
    page_numbers.index(page),  # This is how we render the current page in the first place.
    key='page_selector',
    help=
    "This is just an example page selector, but we don't need to use this as we will drive pages from the sidebar.",
    on_change=callback)
######

# Since the page re-renders every time the selectbox changes, we'll always have the latest page out
# of the query params.
module_str = 'pages.page%s' % page
print(module_str)

# check if the page exists
spec = importlib.util.find_spec(module_str)
if page is None:
  raise Exception("Page %s is was not found" % page)

# Finally import that page's .py file and call it's display function in it.
print("About to render: %s" % module_str)
current_page = importlib.import_module(module_str)
current_page.display()
####################################
