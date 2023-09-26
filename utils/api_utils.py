import streamlit as st
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.auth import create_stub
from clarifai.client.model import Model
from google.protobuf import json_format

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()

def concept_key(concept):
  return "%s (%s)" % (concept.id, concept.name)


def concept_list(userDataObject):
   list_concepts_response = stub.ListConcepts(
    service_pb2.ListConceptsRequest(
       user_app_id=userDataObject)
    )
   if list_concepts_response.status.code != status_code_pb2.SUCCESS:
    print(list_concepts_response.status)
    raise Exception("List concept failed, status: " + list_concepts_response.status.description)

   return list_concepts_response

def list_all_inputs(userDataObject):
  response = stub.ListInputs(
    service_pb2.ListInputsRequest(user_app_id=userDataObject, per_page=16))
  
  return response

def show_error(response, request_name):
  st.error(f"There was an error with your request to {request_name}")
  st.json(json_format.MessageToJson(response, preserving_proto_field_name=True))
  raise Exception(f"There was an error with your request to {request_name} {response.status.description}")
  st.stop()


def get_annotations_for_input_batch(stub, userDataObject, metadata, inputs):

  annotations = []
  if len(inputs) == 0:
    return annotations
  input_ids = [inp.id for inp in inputs]

  list_annotations_response = stub.ListAnnotations(
      service_pb2.ListAnnotationsRequest(
          user_app_id=
          userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
          input_ids=input_ids,
          per_page=1000,  # FIXME(zeiler): this needs proper pagination.
      ),
      metadata=metadata)
  if list_annotations_response.status.code != status_code_pb2.SUCCESS:
    show_error(list_annotations_response,"ListAnnotations")
  for annotation_object in list_annotations_response.annotations:
    if len(annotation_object.data.concepts) > 0 or len(annotation_object.data.regions) > 0:
      # print(annotation_object)
      annotations.append(annotation_object)
    else:
      print("ZZZZZZZZZZZZZZZZZZ")
      print(annotation_object)

  return annotations


def predict_from_image(stub, metadata, img_bytes, user_id, app_id, model_id, version_id):
  '''
      '''
  model_obj = Model(model_id=model_id, user_id=user_id, app_id=app_id)
  if version_id is not None:
    model_obj.model_version.id=version_id

  try:
      response = model_obj.predict_by_bytes(
      img_bytes,
      "image")

  except Exception as e:
      st.error(f"Model predict error : {e} ")
      st.stop()

  #st.write(f"response for predict by image:{response}")
  return response

def predict_from_text(stub, metadata, text, user_id, app_id, model_id, version_id):
  '''
      '''
  #sdk code using predict by bytes
  model_obj = Model(model_id=model_id, user_id=user_id, app_id=app_id)
  if version_id is not None:
    model_obj.model_version.id=version_id

  try:
      response = model_obj.predict_by_bytes(bytes(
      text, 'utf-8'),
      "text")

  except Exception as e:
      st.error(f"Model predict error : {e} ")
      st.stop()

  return response