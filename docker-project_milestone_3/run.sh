#!/bin/bash

echo "TODO: fill in the docker run command"
docker run -it --expose 0.0.0.0:5000:5000/tcp --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:0.0.1