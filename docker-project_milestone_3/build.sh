#!/bin/bash

echo "TODO: fill in the docker build command"
docker build -t ift6758/serving:0.0.1 . -f Dockerfile.serving

docker build -t ift6758/serving:0.0.2 . -f Dockerfile.streamlit

