![Clarifai logo](media/logo.png)

# Clarifai App Mode Gallery


This is a test streamlit app with  the official Clarifai Python utilities. This repo includes higher level convencience classes, functions, and scripts to make using our [API](https://docs.clarifai.com) easier. This is built on top of the [Clarifai Python gRPC Client](https://github.com/Clarifai/clarifai-python-grpc).

* Try the Clarifai demo at: https://clarifai.com/demo
* Sign up for a free account at: https://clarifai.com/signup
* Read the documentation at: https://docs.clarifai.com/


## Installation

First install the [clarifai-utils](https://github.com/Clarifai/clarifai-utils) package following these [instructions](https://github.com/Clarifai/clarifai-utils#installation).

Then git clone and setup this repo:
```cmd
git clone git@github.com:Clarifai/clarifai-mode-gallery.git
cd clarifai-mode-gallery
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
