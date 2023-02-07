import streamlit as st
from clarifai.modules.css import ClarifaiStreamlitCSS

##########################################################
ClarifaiStreamlitCSS.insert_default_css(st)

st.header("Widgets")

st.button('button')
st.download_button('download button', 'download.csv')
st.checkbox('checkbox')
st.radio('radio', ['option 1', 'option 2'])
st.selectbox('selectbox', ['option 1', 'option 2'])
st.multiselect('multiselect', ['option 1', 'option 2'])
st.slider('slider', 1.0, 5.0)
st.select_slider('select slider', range(5))
st.text_input('text input')
st.number_input('number input')
st.text_area('text area')
st.date_input('date input')
st.time_input('time input')
st.file_uploader('file uploader', type='csv')
