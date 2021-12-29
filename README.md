![Clarifai logo](https://www.clarifai.com/hs-fs/hubfs/logo/Clarifai/clarifai-740x150.png?width=240)

# Clarifai App Mode Gallery


This is a test streamlit app with  the official Clarifai Python utilities. This repo includes higher level convencience classes, functions, and scripts to make using our [API](https://docs.clarifai.com) easier. This is built on top of the [Clarifai Python gRPC Client](https://github.com/Clarifai/clarifai-python-grpc).

* Try the Clarifai demo at: https://clarifai.com/demo
* Sign up for a free account at: https://clarifai.com/signup
* Read the documentation at: https://docs.clarifai.com/


## Installation

First install the [clarifai-utils](https://github.com/Clarifai/clarifai-utils) package following these [instructions](https://github.com/Clarifai/clarifai-utils#installation).

Then git clone and setup this repo:
```cmd
git clone git@github.com:Clarifai/clarifai-module-gallery.git
cd clarifai-module-gallery
pip install -r requirements.txt
```

## Versioning

This library doesn't use semantic versioning. The first two version numbers (`X.Y` out of `X.Y.Z`) follow the API (backend) versioning, and
whenever the API gets updated, this library follows it.

The third version number (`Z` out of `X.Y.Z`) is used by this library for any independent releases of library-specific improvements and bug fixes.

## Getting started

After installation you just need to run the streamlit app:
```cmd
streamlit run app.py
```

Find your user_id [here](https://portal.clarifai.com/settings/profile), app_id (of whatever app you want to interact with in your account) and personal access token [here](https://portal.clarifai.com/settings/authentication). 

Put them into the following parts of the url below in your browser:
http://localhost:8501?user_id={user_id}&app_id={app_id}&pat={pat}


### Building Single Page Apps
For a single page app all you need to implement is the app.py file. You're of course free to import any other python modules you build but they will all be used to render that single page. A single page app will still let `page=N` come in as a query param but it will be ignored. 

### Building Multi-Page Apps
The example in this repo shows how you can build a multi-page application. `app.py` essentially looks at the `page=N` query param (where N is a number from 1 to as many pages as you decide to make) and uses that to import the `pages/page{N}.py` as a module and then call that module's `display()` function. Therefore you can implement as many pages as you like with each pages/pageN.py looking something like: 
```python
def display(): 
  # streamlist stuff to be rendered on that page. 
```

An example of jumping between pages using a dropdown is also provided in `app.py`. This is just an example that might make it easier when doing local development of multi-page apps (though you can always use the page=N query param). However, when your app is integrated into the sidebar of our UIs it will receive the `page=N` query param when someone clicks on the navbar link so there is no need for the page dropdown to remain in an app mode once it's ready for production. 

If `page=N` is not provided, the code defaults to `page=1`. 

Note: in the future this page handling will likely be cleaned up as another python pip installable package so that we don't have to copy and paste it. 

## Using Clarifai CSS Styles

This repo includes a style.css file that renders many (not all) of the streamlit widgets using Clarifai's styles. The way it works is it should be loaded (see `local_css`) at the top of your streamlit `app.py` in order to inject the styles into the rendered html page. Eventually we plan to fully host this style file and load it remotely from that url so that it's always the most up to date style file. 

### If you've already created an app

You an copy the style.css file from this repo into your repo and then add the following code snippet to get the styles loaded on render: 
```python
def local_css(file_name):
  with open(file_name) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")
```






