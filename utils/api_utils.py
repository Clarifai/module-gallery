from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
import streamlit as st


def concept_key(concept):
  return "%s (%s)" % (concept.id, concept.name)

def show_error(response, request_name):
  st.error("There was an error with your request to "+request_name)
  st.json(json_format.MessageToJson(response, preserving_proto_field_name=True))
  raise Exception("There was an error with your request to "+request_name+" "+response)
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
  request = service_pb2.PostModelOutputsRequest(
      user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
      # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
      model_id=model_id,
      inputs=[
          resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(base64=img_bytes)))
      ])
  if version_id is not None:
    request.version_id = version_id

  response = stub.PostModelOutputs(request, metadata=metadata)
  # print(response)
  if response.status.code != status_code_pb2.SUCCESS:
    show_error(response,"PostModelOutputs")

  return response

def predict_from_text(stub, metadata, text, user_id, app_id, model_id, version_id):
  '''
      '''
  request = service_pb2.PostModelOutputsRequest(
      user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
      # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
      model_id=model_id,
      inputs=[
          resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=text)))
      ])
  if version_id is not None:
    request.version_id = version_id

  response = stub.PostModelOutputs(request, metadata=metadata)
  # print(response)
  if response.status.code != status_code_pb2.SUCCESS:
    show_error(response,"PostModelOutputs")

  return response