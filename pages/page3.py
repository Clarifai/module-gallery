import altair as alt
import pandas as pd
import streamlit as st
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
## Import in the Clarifai gRPC based objects needed
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
## Import in the Clarifai gRPC based objects needed
from stqdm import stqdm

from utils.api_utils import concept_key, get_annotations_for_input_batch

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

page_size = 16

lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size=page_size)


def display():
  with st.form(key="metrics-inputs"):
    st.text("This will compute a bunch of stats about your app. You ready?")
    submitted = st.form_submit_button('Ready')

  if submitted:
    # Set in the app.py file.
    total = st.session_state['total']

    concepts = []
    for inp in stqdm(
        lister.concepts_generator(), desc="Listing all the concepts in the app", total=total):
      concepts.append(inp)
    concept_ids = [concept_key(c) for c in concepts]
    print(concept_ids)

    # List all the inputs with a nice tqdm progress bar in the UI.
    all_inputs = []
    for inp in stqdm(
        lister.inputs_generator(), desc="Listing all the inputs in the app", total=total):
      all_inputs.append(inp)

    # Stream inputs from the app
    all_annotations = []
    for i in stqdm(range(0, len(all_inputs), page_size), desc="Loading annotations for inputs"):
      batch = all_inputs[i:i + page_size]
      annotations = get_annotations_for_input_batch(stub, userDataObject, metadata, batch)
      all_annotations.extend(annotations)
    ###################

    # Now we aggregate counts over all the annotations we've loaded.
    # this will be a dict of dicts with outer key being the concept_key and the inner key being the
    # "p" or "n" for positive or negative.

    def accume_concept(counts_by_input_id, counts, c):
      key = concept_key(c)
      counts_by_input_id.setdefault(ann.input_id, {})
      counts_by_input_id[ann.input_id].setdefault(key, {"p": 0, "n": 0})
      counts.setdefault(key, {"p": 0, "n": 0})
      if c.value > 0:
        counts[key]["p"] += 1
        counts_by_input_id[ann.input_id][key]['p'] += 1
      else:
        counts[key]["n"] += 1
        counts_by_input_id[ann.input_id][key]['n'] += 1

    counts_by_input_id = {}  # this has an extra outer key by input_id.
    counts = {}
    for ann in all_annotations:
      for c in ann.data.concepts:
        accume_concept(counts_by_input_id, counts, c)
      for r in ann.data.regions:
        for c in r.data.concepts:
          accume_concept(counts_by_input_id, counts, c)

    print("These are the annotation counts per concept")
    counts_source = pd.DataFrame.from_dict({
        "concept": concept_ids,
        "positives": [counts.get(k, {
            "p": 0
        })["p"] for k in concept_ids],
        "negatives": [counts.get(k, {
            "n": 0
        })["n"] for k in concept_ids],
    })
    print(counts_source)

    counts_melted = counts_source.melt('concept', var_name='posneg', value_name='count')
    print(counts_melted)

    print("These are the input counts per concept")
    # for the counts that are unique per input.
    unique_counts = {}
    for _, d in counts_by_input_id.items():
      for key, input_counts in d.items():
        unique_counts.setdefault(key, {"p": 0, "n": 0})
        if input_counts['p'] > 0:  # if more than 1 for this input we increment by 1.
          unique_counts[key]["p"] += 1
        if input_counts['n'] > 0:
          unique_counts[key]["n"] += 1
    unique_counts_source = pd.DataFrame.from_dict({
        "concept": concept_ids,
        "positives": [unique_counts.get(k, {
            "p": 0
        })["p"] for k in concept_ids],
        "negatives": [unique_counts.get(k, {
            "n": 0
        })["n"] for k in concept_ids],
    })
    print(unique_counts_source)

    unique_counts_melted = unique_counts_source.melt(
        'concept', var_name='posneg', value_name='count')
    print(unique_counts_melted)

    get_input_count_response = st.session_state['get_input_count_response']
    st.header("Input status", anchor="input-status")
    status = pd.DataFrame({
        "cat": ["processed", "processing", "to process", "error processing"],
        "value": [
            get_input_count_response.counts.processed, get_input_count_response.counts.processing,
            get_input_count_response.counts.to_process, get_input_count_response.counts.errors
        ]
    })
    c = alt.Chart(status).mark_bar().encode(x='cat', y='value')
    text = c.mark_text(dy=-5).encode(text='value')
    st.altair_chart(c + text, use_container_width=True)

    base = alt.Chart(status).mark_arc().encode(theta=alt.Theta('value', stack=True), color='cat')
    pie = base.mark_arc(outerRadius=120)
    text = base.mark_text(radius=140, size=20).encode(text="value:N")
    st.altair_chart(pie + text, use_container_width=True)

    # Finally plot the results to the UI.
    st.header("Annotation stats", anchor="data-stats")

    st.text("These are annotation counts per concept.")
    c = alt.Chart(counts_melted).mark_bar().encode(x='concept', y='sum(count)', color='posneg')
    st.altair_chart(c, use_container_width=True)

    st.text('someting else')

    base = alt.Chart(counts_melted).mark_arc().encode(
        theta=alt.Theta('sum(count)', stack=True), color='concept')
    pie = base.mark_arc(outerRadius=120)
    text = base.mark_text(radius=140, size=20).encode(text="concept")
    st.altair_chart(pie + text, use_container_width=True)

    st.text("These are input counts per concept.")
    c = alt.Chart(unique_counts_melted).mark_bar().encode(
        x='concept', y='sum(count)', color='posneg')
    st.altair_chart(c, use_container_width=True)

    base = alt.Chart(unique_counts_melted).mark_arc().encode(
        theta=alt.Theta('sum(count)', stack=True), color='concept')
    pie = base.mark_arc(outerRadius=120)
    text = base.mark_text(radius=160, size=20).encode(text="concept")
    st.altair_chart(pie + text, use_container_width=True)