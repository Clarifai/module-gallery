import streamlit as st
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from clarifai_utils.modules.css import ClarifaiStreamlitCSS
from clarifai_utils.modules.pages import ClarifaiModulePageManager

########################
# Required in every Clarifai streamlit app
########################

st.set_page_config(layout="wide")

ClarifaiStreamlitCSS.insert_default_css(st)

# Validate and parse the query params we need.
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = auth.get_stub()
########################
metadata = auth.metadata
userDataObject = auth.get_user_app_id_proto()

page_size = 3

lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size)

###################

# Shared API calls before any page renders. This can be used to save state for a particular browser
# session. I'm using this here to show an example of how when the app loads we can get a value like
# the counts of inputs, and then use it within the various pages of the multi-page app.
# It's not clear when the session state ever gets cleared or where it is stored.
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
# Get the current page to render
####################################
# Read the page param from the query bar.
qp = st.experimental_get_query_params()

page_manager = ClarifaiModulePageManager()
page = page_manager.get_page_from_query_params(qp)
page_names = page_manager.get_page_names()


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
    page_names,
    page_names.index(page),  # This is how we render the current page in the first place.
    key='page_selector',
    help=
    "This is just an example page selector, but we don't need to use this as we will drive pages from the sidebar.",
    on_change=callback)
######

# Finally import that page's .py file and call it's display function in it.
page_manager.render_page(page)
####################################
