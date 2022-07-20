import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.modules.css import ClarifaiStreamlitCSS
from google.protobuf import json_format
from PIL import Image

from utils.api_utils import init_session_state, predict_from_image

##########################################################
ClarifaiStreamlitCSS.insert_default_css(st)

# This must be within the display() function.
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = auth.get_stub()
metadata = auth.metadata
userDataObject = auth.get_user_app_id_proto()
st.title("Compare Two Classification Models")

init_session_state(st, auth)

st.write("This is a Streamlit app that predict objects and semantic concepts within an image.")

st.header("Step 2: Upload and Predict Objects in Image")
file_data = st.file_uploader("Select an image", type=['jpg', 'jpeg', 'png'])

if file_data == None:
  st.warning("File not uploaded.")
  st.stop()
else:
  st.write("File Uploaded!")
# ToDo(Andrew:) Add a way to export concepts/predictions to
img_b = file_data.getvalue()
image = Image.open(file_data)
st.header("Upload Preview")
st.image(image, width=400)

with st.form(key="model-compare"):

  st.header("Step 3: Upload Model ID and Version ID")
  cols = st.columns(2)
  cols[0].text("Model 1")
  USER_ID1 = cols[0].text_input(" Paste user_id, then press enter:", key=1)
  APP_ID1 = cols[0].text_input(" Paste app_id, then press enter:", key=2)
  MODEL_ID1 = cols[0].text_input(" Paste Model ID, then press enter:", key=3)
  VERSION_ID1 = cols[0].text_input(" Paste VERSION_ID, then press enter:", key=4)
  cols[1].text("Model 2")
  USER_ID2 = cols[1].text_input(" Paste user_id, then press enter:", key=5)
  APP_ID2 = cols[1].text_input(" Paste app_id, then press enter:", key=6)
  MODEL_ID2 = cols[1].text_input(" Paste Model ID, then press enter:", key=7)
  VERSION_ID2 = cols[1].text_input(" Paste VERSION_ID, then press enter:", key=8)

  submitted = st.form_submit_button('Compare Predicts')

if submitted:  # this blocks here until the person hits the submit button.
  # NOTE(zeiler): not sure if this is the best way to validate a form.
  for var in [
      USER_ID1, USER_ID2, APP_ID1, APP_ID2, MODEL_ID1, MODEL_ID2, VERSION_ID1, VERSION_ID2
  ]:
    if var == '':
      st.error("You must provide all the fields in the form before submitting.")
      st.stop()
  response1 = predict_from_image(stub, auth.metadata, img_b, USER_ID1, APP_ID1, MODEL_ID1,
                                 VERSION_ID1)
  response2 = predict_from_image(stub, auth.metadata, img_b, USER_ID2, APP_ID2, MODEL_ID2,
                                 VERSION_ID2)

  json_string1 = json_format.MessageToJson(response1, preserving_proto_field_name=True)
  json_string2 = json_format.MessageToJson(response1, preserving_proto_field_name=True)

  # Note(zeiler): these are pretty big and I don't see a default collapsed option.
  # st.header("Responses")
  # st.json(json_string1)
  # st.json(json_string2)

  st.header("Predicted Concepts")
  cols = st.columns(2)
  concept_names = []
  concept_confidences = []
  for concept in response1.outputs[0].data.concepts:
    # st.write('%12s: %.2f' % (concept.name, concept.value))
    concept_names.append(concept.name)
    concept_confidences.append(concept.value)
  df = pd.DataFrame({
      'Concept Names': concept_names,
      'Confidence of Concept': concept_confidences,
  })
  cols[0].text("Model 1 Predictions")
  cols[0].write(df)

  concept_names = []
  concept_confidences = []
  for concept in response2.outputs[0].data.concepts:
    # st.write('%12s: %.2f' % (concept.name, concept.value))
    concept_names.append(concept.name)
    concept_confidences.append(concept.value)
  df2 = pd.DataFrame({
      'Concept Names': concept_names,
      'Confidence of Concept': concept_confidences,
  })
  cols[1].text("Model 2 Predictions")
  cols[1].write(df2)

  df3 = pd.merge(df, df2, how="outer", on="Concept Names")
  df3 = df3.rename(columns={
      "Confidence of Concept_x": "Model 1",
      "Confidence of Concept_y": "Model 2"
  })

  cols = st.columns(2)
  c = alt.Chart(df3).mark_bar().encode(y='Concept Names', x='Model 1')
  text = c.mark_text(dx=3).encode(text='Model 1')
  cols[0].altair_chart(c + text, use_container_width=True)
  c = alt.Chart(df3).mark_bar().encode(y='Concept Names', x='Model 2')
  text = c.mark_text(dx=3).encode(text='Model 2')
  cols[1].altair_chart(c + text, use_container_width=True)

  print(df3)
  st.text("Model Predictions Compared")

  def max_color(s, props=''):
    return np.where(s == np.nanmax(s.values[1:]), props, '')

  def nan_color(s, props=''):
    return np.where(s.isnull(), props, '')

  # Style the cells before writing it.
  stdf3 = df3.style.apply(max_color, props='color:white;background-color:green', axis=1)
  stdf3 = stdf3.apply(nan_color, props='color:black;background-color:black', axis=None)
  st.write(stdf3)
  cmelted = df3.melt('Concept Names', var_name='model', value_name='confidence')
  print(cmelted)

  c = alt.Chart(cmelted).mark_bar().encode(
      y='model', x='sum(confidence)', color='model', row='Concept Names')
  st.altair_chart(c)  #, use_container_width=True)
