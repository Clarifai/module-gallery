
import streamlit as st
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from clarifai_utils.modules.css import ClarifaiStreamlitCSS
from stqdm import stqdm

from annotated_text import annotated_text

from utils.api_utils import concept_key, get_annotations_for_input_batch, init_session_state

page_size = 16

def display():

  ClarifaiStreamlitCSS.insert_default_css(st)

  # This must be within the display() function.
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = auth.get_stub()
  metadata = auth.metadata
  userDataObject = auth.get_user_app_id_proto()
  lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size=page_size)

  init_session_state(st, auth)

  st.title("NER Annotation")
  with st.form(key="ner-inputs"):    
    # vvvv - Note: the model list can be from a local CSV or hard coded - not sure there is a way to grab all NER models
    #        Can also have a dict or tuples of model name / model ids if needed
    model = st.selectbox(
      'Model to use for NER',
      ('Model 1', 'Model 2', 'Model 3'))
    annot_text = st.text_input("Text to annotate")
    submitted = st.form_submit_button('Generate')

  if submitted:
    print("Submitted!")
    # TODO: Make request with API
    #  model to call is in the variable model
    #  text to call with is in the variable annot_text

    # TODO: Add output here after making request
    #       Example formatting:
    #       Tuples of (word, part of speech, color)
    #       Unannotated can be passed in as string (no tuple)
    #
    #   annotated_text(
    #     "This ",
    #     ("is", "verb", "#8ef"),
    #     " some ",
    #     ("annotated", "adj", "#faa"),
    #     ("text", "noun", "#afa"),
    #     " for those of ",
    #     ("you", "pronoun", "#fea"),
    #     " who ",
    #     ("like", "verb", "#8ef"),
    #     " this sort of ",
    #     ("thing", "noun", "#afa"),
    # )
    

    
    




if __name__ == '__main__':
  display()