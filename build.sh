#!/bin/bash

mkdir -p .npm

docker stop ituna-docs

docker build -t ituna-docs .

docker run --rm \
    --name ituna-docs \
    -p 127.0.0.1:8000:8000 \
    -u $(id -u):$(id -g) \
    -v $(pwd):/app \
    -v $(pwd)/.npm:/.npm \
    -w /app \
    -it \
    ituna-docs \
    ./entrypoint.sh
