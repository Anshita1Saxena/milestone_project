#!/bin/bash

echo "TODO: fill in the docker run command"
docker run -it --expose 127.0.0.1:8890:8890/tcp --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:0.0.1 
