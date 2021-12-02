import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

packages = setuptools.find_packages(include=["*"])

setuptools.setup(
    name="clarifai-mode-gallery",
    version="0.0.1",
    author="Clarifai",
    author_email="support@clarifai.com",
    description="Clarifai App Mode Gallery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clarifai/clarifai-mode-gallery",
    packages=packages,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires='>=3.6',
    install_requires=[
        "clarifai-utils>=0.0.1",
    ],
    package_data={p: ["*.pyi"]
                  for p in packages},
    include_package_data=True)
