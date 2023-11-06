import streamlit as st
from clarifai.client.app import App
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.input import Inputs
from google.protobuf import json_format

auth = ClarifaiAuthHelper.from_streamlit(st)
userDataObject = auth.get_user_app_id_proto()


def concept_key(concept):
  return "%s (%s)" % (concept.id, concept.name)


def concept_list(user_id, app_id):
  app_obj = App(app_id=app_id, user_id=user_id)
  try:
    list_concepts_response = list(app_obj.list_concepts(per_page=100))
    return list_concepts_response

  except Exception as e:
    st.error(f"List concept failed, status: {e}")


def list_all_inputs(user_id, app_id):
  input_obj = Inputs(user_id=user_id, app_id=app_id)
  try:
    response = input_obj.list_inputs(
        input_type="image", per_page=100
    )  #Per_page is set to 100 since no of images to display in the portal is limited to 100
    return response
  except Exception as e:
    st.error(f"There was an error with your request to list inputs: {e}")


def show_error(response, request_name):
  st.error(f"There was an error with your request to {request_name}")
  st.json(json_format.MessageToJson(response, preserving_proto_field_name=True))
  raise Exception(
      f"There was an error with your request to {request_name} {response.status.description}")
  st.stop()


def get_annotations_for_input_batch(userDataObject, inputs):

  annotations = []
  if len(inputs) == 0:
    return annotations
  input_obj = Inputs(user_id=userDataObject.user_id, app_id=userDataObject.app_id)
  try:
    list_annotations_response = list(input_obj.list_annotations(batch_input=inputs))
  except Exception as e:
    st.error(f"There was an error with your request to list annotations: {e}")
    st.stop()

  for annotation_object in list_annotations_response:
    if len(annotation_object.data.concepts) > 0 or len(annotation_object.data.regions) > 0:
      # print(annotation_object)
      annotations.append(annotation_object)

  return annotations
