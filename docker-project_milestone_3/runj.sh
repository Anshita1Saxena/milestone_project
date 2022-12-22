#!/bin/bash
#docker run -it --expose 0.0.0.0:5000:8891/tcp --env COMET_API_KEY=$COMET_API_KEY ift6758/serving:0.0.1
waitress-serve --listen=0.0.0.0:5000 --port=5000 app:app