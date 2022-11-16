import altair as alt
import pandas as pd
import streamlit as st
from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_utils.auth.helper import ClarifaiAuthHelper
from clarifai_utils.listing.lister import ClarifaiResourceLister
from clarifai_utils.modules.css import ClarifaiStreamlitCSS
from stqdm import stqdm
from random import sample
from time import perf_counter_ns, time, localtime
import numpy as np
from scipy.stats import norm


from utils.api_utils import concept_key, get_annotations_for_input_batch, init_session_state

page_size = 100

def display():

  ClarifaiStreamlitCSS.insert_default_css(st)

  # This must be within the display() function.
  auth = ClarifaiAuthHelper.from_streamlit(st)
  stub = auth.get_stub()
  metadata = auth.metadata
  userDataObject = auth.get_user_app_id_proto()
  lister = ClarifaiResourceLister(stub, metadata, auth.user_id, auth.app_id, page_size=page_size)

  init_session_state(st, auth)
  # SET SESSION VARIABLES
  st.session_state['models'] = st.session_state['models'] if 'models' in st.session_state else list()
  st.session_state['model_ids'] = st.session_state['model_ids'] if 'model_ids' in st.session_state else list()
  st.session_state['datasets'] = st.session_state['datasets'] if 'datasets' in st.session_state else list()
  st.session_state['dataset_ids'] = st.session_state['dataset_ids'] if 'dataset_ids' in st.session_state else list()

  if not st.session_state['models']:
    st.session_state['models'] = [m for m in lister.models_generator(only_in_app=False)]

  if not st.session_state['model_ids']:
    st.session_state['model_ids'] = [m.id for m in st.session_state['models'] if m.app_id == auth.app_id]

  if not st.session_state['datasets']:
    st.session_state['datasets'] = [inp for inp in lister.datasets_generator()]

  if not st.session_state['dataset_ids']:
    st.session_state['dataset_ids'] = [inp.id for inp in st.session_state['datasets']]

  st.title("Platform Benchmarking")
  st.text("This module allows a user to select a model within the application and run general benchmarks on it.")
  st.subheader("Application: {}".format(auth.app_id))

  def include_clarifai(app_id):
    '''
    Changes the model options depending on if the Include Clarifai Models checkbox is toggled.
    '''
    if st.session_state['include_clarifai_models']:
      st.session_state['model_ids'] = [m.id for m in st.session_state['models']]
    else:
      st.session_state['model_ids'] = [m.id for m in st.session_state['models'] if m.app_id == app_id]
  
  # Need to place this outside of the streamlit form in order to have the callback work.
  st.checkbox("Include Clarifai Models", on_change=include_clarifai, args=(auth.app_id, ), key='include_clarifai_models', 
              help="Checking this will add all accessible Clarifai models to the list of selectable models.")

  with st.form(key="benchmarks-inputs"):
    form_col1, form_col2 = st.columns([1,2], gap="large")
    with form_col1:
      model_option = st.selectbox(
        'Which model would you like to benchmark?',
        options=[None] + st.session_state['model_ids'],
        key='model_option_box',
        index=0
      )

      dataset_option = st.selectbox(
        'Which dataset would you like to benchmark with?',
        options=[None] + st.session_state['dataset_ids'],
        key='dataset_option_box',
        index=0
      )
    
    with form_col2:
      st.checkbox("Warm Model", key="warm_model", help="Che")
      max_inputs = st.slider('Maximum inputs to use', 30, 500, 50)
      # output_bins = st.slider('Maximum binning for visualization', 5, 100, 20)

    submitted = st.form_submit_button('Ready')

  if submitted:
    if model_option is None or dataset_option is None:
      print("You need to select a model and a dataset to continue ")
      return
    # Benchmarks to generate
    #
    # 1. Inference Requests against the model from the dataset (time)
    # 2. Visual Search
    # 3. Annotation?  (Not model specific)
    # 4. 

    # Get list of inputs from dataset
    all_inputs = []
    for inp in stqdm(
        lister.dataset_inputs_generator(dataset_id=dataset_option), desc="Fetching all the inputs in the dataset"):
      all_inputs.append(inp)

    st.text("Loaded a total of {} inputs.".format(len(all_inputs)))
    inputs_to_use = all_inputs
    if len(all_inputs) > max_inputs:
      st.text("{} will be used.".format(max_inputs))
      inputs_to_use = sample(all_inputs, max_inputs)

    time_data = {'Latency': [], 'Time': []}
    for inp in stqdm(
      inputs_to_use, desc="Making prediction requests.", total=len(inputs_to_use)):
      # call_time = time()
      # localTime = localtime(time())
      # lt = "{}:{}:{}".format(localTime.tm_hour,localTime.tm_min,localTime.tm_sec)

      start = perf_counter_ns()
      post_model_outputs_response = stub.PostModelOutputs(
          service_pb2.PostModelOutputsRequest(
              user_app_id=userDataObject,
              model_id=model_option,
              inputs=[
                  resources_pb2.Input(
                      data=resources_pb2.Data(
                          image=resources_pb2.Image(
                              url=inp.data.image.url
                          )
                      )
                  )
              ]
          ),
          metadata=metadata
      )
      end = perf_counter_ns()
      
      if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

      # time_list.append(((end - start) / 1000000.))  # put in milliseconds
      time_data['Latency'].append(((end - start) / 1000000.))
      time_data['Time'].append(start)
    
    p90latency = np.percentile(time_data['Latency'], 90)
    p95latency = np.percentile(time_data['Latency'], 95)
    p99latency = np.percentile(time_data['Latency'], 99)
    print("P90: {:.2f}".format(p90latency))
    print("P95: {:.2f}".format(p95latency))
    print("P99: {:.2f}".format(p99latency))

    # time_data = {'Latency': time_list}
    population_sample = pd.DataFrame(data=time_data)

    st.header("Inference Latency on {}".format(model_option), anchor="inference-latency")
    
    
    col1, col2 = st.columns([1, 3], gap="large")
    with col1:
      #
      # Draw the PXX Table for concrete values
      # TODO: Styling isnt working properly
      cell_hover = {  # for row hover use <tr> instead of <td>
          'selector': 'td:hover',
          'props': [('background-color', '#ffffb3')]
      }
      row_colors = [
        {'selector': '.p90', 'props': 'background-color: #e89b58;'},
        {'selector': '.p95', 'props': 'background-color: #e64040;'},
        {'selector': '.p99', 'props': 'background-color: #a940e6;'},
      ]
      df = pd.DataFrame(
        [p90latency,p95latency,p99latency]
      )
      df.columns = ["Latency (ms)"]
      df.index = ["P90","P95","P99"]
      
      cell_colors = pd.DataFrame([['p90'],
                                  ['p95'],
                                  ['p99']],
                                  index=df.index,
                                  columns=df.columns)
      
      df.style.set_td_classes(cell_colors)
      df.style.format('{:.2f')
      df.style.set_table_styles([cell_hover, *row_colors])
      st.subheader("Latency Values")
      st.table(df)
      st.caption('Percentile latency measurements. P90 is the latency where 90\% of requests are the reported time or lower.  P95 is where 95\% of requests are lower than this time, and P99 represents the limit where 99\% of requests fall under.')

      # Used for modeling a gaussian distribution (Central Limit Theorum)
      sample_mean = np.mean(population_sample['Latency'])
      sample_std = np.std(population_sample['Latency'])
      x = population_sample["Latency"].sort_values()

      normal_dist = pd.DataFrame({'Latency': x, 'PDF': norm.pdf(x, loc=sample_mean, scale=sample_std)})
      normal_dist_visual = alt.Chart(normal_dist).mark_line().encode(
          x=alt.X('Latency:Q', axis=alt.Axis(labelAngle=-45)),
          y='PDF'
      )

      st.subheader("System Latency Model")
      st.altair_chart(normal_dist_visual, use_container_width=False)
      st.caption("Model of system latency expected under current conditions for this model.  Shown as a distribution with latency on the x axis and probability shown on the y.")
      
    with col2:
      # Set up the Latency chart
      base = alt.Chart(population_sample)
      # histogram = base.mark_bar().encode(x=alt.X('Latency:Q', bin=alt.Bin(maxbins=output_bins), axis=alt.Axis(labels=True)), y='count()')
      line_chart = base.mark_line().encode(y='Latency:Q', x=alt.X('Time:N'))

      p90base = alt.Chart(pd.DataFrame({'p90': [p90latency], 'name': ['P90']}))
      p90 = p90base.mark_rule(color='orange').encode(y='p90:Q', size=alt.value(3))
      p95base = alt.Chart(pd.DataFrame({'p95': [p95latency], 'name': ['P95']}))
      p95 = p95base.mark_rule(color='red').encode(y='p95:Q', size=alt.value(3))
      p99base = alt.Chart(pd.DataFrame({'p99': [p99latency], 'name': ['P99']}))
      p99 = p99base.mark_rule(color='purple').encode(y='p99:Q', size=alt.value(3))
      
      # text = base.mark_text(dy=-5).encode(text='value')
      # p90text = p90base.mark_text(baseline='top').encode(text="name", x='test')
      
      # st.altair_chart(line_chart + p90 + p95 + p99 + p90text, use_container_width=True)
      st.subheader("Response Times")
      st.altair_chart(line_chart + p90 + p95 + p99, use_container_width=True)
      st.caption("Response times for inference requests against the model.  P90 threshold is shown in orange, P95 threshold is shown in red, and P99 threshold is shown in purple.")
    # base = alt.Chart(status).mark_arc().encode(theta=alt.Theta('value', stack=True), color='cat')
    # pie = base.mark_arc(outerRadius=120)
    # text = base.mark_text(radius=140, size=20).encode(text="value:N")
    # st.altair_chart(pie + text, use_container_width=True)

    # # Finally plot the results to the UI.
    # st.header("Annotation stats", anchor="data-stats")

    # st.text("These are annotation counts per concept.")
    # c = alt.Chart(counts_melted).mark_bar().encode(x='concept', y='sum(count)', color='posneg')
    # st.altair_chart(c, use_container_width=True)

    # st.text('someting else')

    # base = alt.Chart(counts_melted).mark_arc().encode(
    #     theta=alt.Theta('sum(count)', stack=True), color='concept')
    # pie = base.mark_arc(outerRadius=120)
    # text = base.mark_text(radius=140, size=20).encode(text="concept")
    # st.altair_chart(pie + text, use_container_width=True)

    # st.text("These are input counts per concept.")
    # c = alt.Chart(unique_counts_melted).mark_bar().encode(
    #     x='concept', y='sum(count)', color='posneg')
    # st.altair_chart(c, use_container_width=True)

    # base = alt.Chart(unique_counts_melted).mark_arc().encode(
    #     theta=alt.Theta('sum(count)', stack=True), color='concept')
    # pie = base.mark_arc(outerRadius=120)
    # text = base.mark_text(radius=160, size=20).encode(text="concept")
    # st.altair_chart(pie + text, use_container_width=True)


if __name__ == '__main__':
  display()
