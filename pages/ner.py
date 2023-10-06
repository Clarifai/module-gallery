import streamlit as st
from annotated_text import annotated_text
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper

from utils.api_utils import predict_from_text

#Note(Eran) turn to multipage app where all demos are nested under a pretrained_models_demos subfolder

# Uses the enviorment variables CLARIFAI_USER_ID, CLARIFAI_APP_ID, CLARIFAI_PAT
# (although only PAT is needed for authentication as we're used a pretrained model)
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
metadata = auth.metadata

# Fixed arguments to use NER-English model
user_id = 'clarifai'
user_app_id = 'main'
model_id = 'ner-english'
model_version_id = 'a813ff5b362c41f790c506b871e7dea4'

# Main page title
st.title("NER Demo")
# Define color templates
ent_colors = {"PER": "#8ce59e", "LOC": '#8edce4', "ORG": '#79abdc', "MISC": '#8d7dca'}

# Get User text and run prediction
with st.spinner("Finding Entities"):
  text_input = st.text_area("Press ⌘+Enter to rerun NER prediction:",
                            "Eran lives in New York\nand works for Clarifai")

  response = predict_from_text(text_input, user_id, user_app_id, model_id, model_version_id)
  output = response.outputs[0]  #no batching

  # Turn Clarifai response to streamlit annotated text standard
  ents = []
  for region in output.data.regions:
    char_start = region.region_info.span.char_start
    char_end = region.region_info.span.char_end
    entity_name = region.data.concepts[0].name
    entity_confidence = region.data.concepts[0].value
    if entity_name == "O":
      ents.append(text_input[char_start:char_end])
    else:
      ents.append((text_input[char_start:char_end], entity_name, ent_colors[entity_name]))

  annotated_text(*ents)
