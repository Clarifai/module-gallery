from itertools import cycle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
## Import in the Clarifai gRPC based objects needed
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from google.protobuf import json_format
from PIL import Image
from stqdm import stqdm
from vega_datasets import data

from utils.mosaic import download_urls, urls_to_mosaic

########################
# Required in every Clarifai streamlit app
########################
# Validate and parse the query params we need.
try:
  qp = ClarifaiAuthHelper.from_streamlit_query_params(st.experimental_get_query_params())
except:
  qp = ClarifaiAuthHelper.from_env()

st.set_page_config(layout="wide")
########################

# Create the clarifai grpc client.
channel = ClarifaiChannel.get_grpc_channel(base="api.clarifai.com")
stub = service_pb2_grpc.V2Stub(channel)
metadata = qp.metadata
print(metadata)
userDataObject = qp.get_user_app_id_proto()
print(userDataObject)

page_size = 3

lister = ClarifaiResourceLister(stub, metadata, qp.user_id, qp.app_id, page_size)

###################


def concept_key(concept):
  return "%s (%s)" % (concept.id, concept.name)


def get_annotations_for_input_batch(inputs):

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
    print("There was an error with your request!")
    print("\tCode: {}".format(list_annotations_response.outputs[0].status.code))
    print("\tDescription: {}".format(list_annotations_response.outputs[0].status.description))
    print("\tDetails: {}".format(list_annotations_response.outputs[0].status.details))
    raise Exception("List annotations failed, status: " +
                    list_annotations_response.status.description)
  for annotation_object in list_annotations_response.annotations:
    if len(annotation_object.data.concepts) > 0 or len(annotation_object.data.regions) > 0:
      # print(annotation_object)
      annotations.append(annotation_object)
    else:
      print("ZZZZZZZZZZZZZZZZZZ")
      print(annotation_object)

  return annotations


def predict_from_image(img_bytes, user_id, app_id, model_id, version_id):
  '''
      '''
  request = service_pb2.PostModelOutputsRequest(
      user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
      # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
      model_id=model_id,
      version_id=version_id,
      inputs=[
          resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(base64=img_bytes)))
      ])
  response = stub.PostModelOutputs(request, metadata=qp.metadata)
  # print(response)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelOutputs request failed: %r" % response)

  return response


# Shared API calls before anything goes
get_input_count_response = stub.GetInputCount(
    service_pb2.GetInputCountRequest(user_app_id=userDataObject), metadata=metadata)
if get_input_count_response.status.code != status_code_pb2.SUCCESS:
  raise Exception("Get input count failed, response: %r" % get_input_count_response)
counts = get_input_count_response.counts
print(counts)

total = get_input_count_response.counts.processed + get_input_count_response.counts.errors + get_input_count_response.counts.errors + get_input_count_response.counts.to_process
print(total)

with st.expander("Metrics mode", expanded=True):

  with st.form(key="metrics-inputs"):
    st.text("This will compute a bunch of stats about your app. You ready?")
    submitted = st.form_submit_button('Ready')
  if submitted:
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
      annotations = get_annotations_for_input_batch(batch)
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

##########################################################
with st.expander("Mosaic", expanded=False):
  ##########################################################
  with st.form(key="mosiac-inputs"):
    mtotal = st.number_input(
        "Select number of images to add to mosaic:", min_value=1, max_value=100)
    submitted = st.form_submit_button('Submit')

  if submitted:
    if mtotal is None or mtotal == 0:
      st.warning("Number of images must be provided.")
      st.stop()
    else:
      st.write("Mosaic number of images will be: {}".format(mtotal))

    # Stream inputs from the app
    all_images = []
    for inp in stqdm(
        lister.inputs_generator(), desc="Listing all the inputs in the app", total=total):
      if inp.data.image is not None:
        all_images.append(inp.data.image)
      if len(all_images) >= mtotal:
        break

    print(all_images)
    url_list = [im.url for im in all_images if im.url != ""]

    mosaic = urls_to_mosaic(url_list)

    st.image(mosaic)
##########################################################

##########################################################
with st.expander("Data Viewer", expanded=False):
  ##########################################################
  with st.form(key="data-inputs"):
    mtotal = st.number_input("Select number of images to view:", min_value=1, max_value=100)
    submitted = st.form_submit_button('Submit')

  if submitted:
    if mtotal is None or mtotal == 0:
      st.warning("Number of images must be provided.")
      st.stop()
    else:
      st.write("Mosaic number of images will be: {}".format(mtotal))

    # Stream inputs from the app
    all_images = []
    for inp in stqdm(
        lister.inputs_generator(), desc="Listing all the inputs in the app", total=total):
      if inp.data.image is not None:
        all_images.append(inp.data.image)
      if len(all_images) >= mtotal:
        break

    url_list = [im.url for im in all_images if im.url != ""]

    filteredImages = [tup[1] for tup in download_urls(url_list)]

    print(filteredImages)
    # caption = [] # your caption here
    cols = cycle(
        st.columns(4))  # st.columns here since it is out of beta at the time I'm writing this
    for idx, filteredImage in enumerate(filteredImages):
      next(cols).image(np.array(filteredImage), use_column_width=True)

##########################################################

with st.expander("Model Compare", expanded=False):
  ##########################################################
  st.title("Compare Two Classification Models")
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
    response1 = predict_from_image(img_b, USER_ID1, APP_ID1, MODEL_ID1, VERSION_ID1)
    response2 = predict_from_image(img_b, USER_ID2, APP_ID2, MODEL_ID2, VERSION_ID2)

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

with st.expander("Other Example Altair Plots", expanded=False):

  # Generating Data
  source = pd.DataFrame({
      'Trial A': np.random.normal(0, 0.8, 1000),
      'Trial B': np.random.normal(-2, 1, 1000),
      'Trial C': np.random.normal(3, 2, 1000)
  })

  c = alt.Chart(source).transform_fold(
      ['Trial A', 'Trial B', 'Trial C'], as_=['Experiment', 'Measurement']).mark_bar(
          opacity=0.3, binSpacing=0).encode(
              alt.X('Measurement:Q', bin=alt.Bin(maxbins=100)),
              alt.Y('count()', stack=None), alt.Color('Experiment:N'))
  print(source)
  st.altair_chart(c)

  source = data.movies.url

  pts = alt.selection(type="single", encodings=['x'])

  rect = alt.Chart(data.movies.url).mark_rect().encode(
      alt.X('IMDB_Rating:Q', bin=True),
      alt.Y('Rotten_Tomatoes_Rating:Q', bin=True),
      alt.Color(
          'count()', scale=alt.Scale(scheme='greenblue'),
          legend=alt.Legend(title='Total Records')))

  circ = rect.mark_point().encode(
      alt.ColorValue('grey'),
      alt.Size('count()', legend=alt.Legend(title='Records in Selection'))).transform_filter(pts)

  bar = alt.Chart(source).mark_bar().encode(
      x='Major_Genre:N',
      y='count()',
      color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))).properties(
          width=550, height=200).add_selection(pts)

  st.altair_chart(
      alt.vconcat(rect + circ, bar).resolve_legend(color="independent", size="independent"))

  source = data.cars()

  st.altair_chart(
      alt.Chart(source).mark_circle().encode(
          alt.X(alt.repeat("column"), type='quantitative'),
          alt.Y(alt.repeat("row"), type='quantitative'),
          color='Origin:N').properties(width=150, height=150).repeat(
              row=['Horsepower', 'Acceleration', 'Miles_per_Gallon'],
              column=['Miles_per_Gallon', 'Acceleration', 'Horsepower']).interactive())

  source = data.cars()

  # Configure the options common to all layers
  brush = alt.selection(type='interval')
  base = alt.Chart(source).add_selection(brush)

  # Configure the points
  points = base.mark_point().encode(
      x=alt.X('Miles_per_Gallon', title=''),
      y=alt.Y('Horsepower', title=''),
      color=alt.condition(brush, 'Origin', alt.value('grey')))

  # Configure the ticks
  tick_axis = alt.Axis(labels=False, domain=False, ticks=False)

  x_ticks = base.mark_tick().encode(
      alt.X('Miles_per_Gallon', axis=tick_axis),
      alt.Y('Origin', title='', axis=tick_axis),
      color=alt.condition(brush, 'Origin', alt.value('lightgrey')))

  y_ticks = base.mark_tick().encode(
      alt.X('Origin', title='', axis=tick_axis),
      alt.Y('Horsepower', axis=tick_axis),
      color=alt.condition(brush, 'Origin', alt.value('lightgrey')))

  # Build the chart
  st.altair_chart(y_ticks | (points & x_ticks))

  source = data.stocks()

  st.altair_chart(
      alt.Chart(source).transform_filter('datum.symbol==="GOOG"').mark_area(
          line={
              'color': 'darkgreen'
          },
          color=alt.Gradient(
              gradient='linear',
              stops=[
                  alt.GradientStop(color='white', offset=0),
                  alt.GradientStop(color='darkgreen', offset=1)
              ],
              x1=1,
              x2=1,
              y1=1,
              y2=0)).encode(alt.X('date:T'), alt.Y('price:Q')))

  source = data.unemployment_across_industries.url

  base = alt.Chart(source).mark_area(
      color='goldenrod', opacity=0.3).encode(
          x='yearmonth(date):T',
          y='sum(count):Q',
      )

  brush = alt.selection_interval(encodings=['x'], empty='all')
  background = base.add_selection(brush)
  selected = base.transform_filter(brush).mark_area(color='goldenrod')

  st.altair_chart(background + selected)

  airports = data.airports.url
  states = alt.topo_feature(data.us_10m.url, feature='states')

  # US states background
  background = alt.Chart(states).mark_geoshape(
      fill='lightgray', stroke='white').properties(
          width=500, height=300).project('albersUsa')

  # airport positions on background
  points = alt.Chart(airports).transform_aggregate(
      latitude='mean(latitude)', longitude='mean(longitude)', count='count()',
      groupby=['state']).mark_circle().encode(
          longitude='longitude:Q',
          latitude='latitude:Q',
          size=alt.Size('count:Q', title='Number of Airports'),
          color=alt.value('steelblue'),
          tooltip=['state:N', 'count:Q']).properties(title='Number of airports in US')

  st.altair_chart(background + points)
