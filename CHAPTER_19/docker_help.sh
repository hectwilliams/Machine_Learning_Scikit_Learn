#!/bin/bash

gcloud=${HOME}/google-cloud-sdk/bin/gcloud
export PROJECT_ID=$(${gcloud} config list project --format "value(core.project)")
export REPO_NAME=secondrepo
export IMAGE_NAME=mytest
export IMAGE_TAG=40
export REGISTRY_ID=us-docker.pkg.dev
export IMAGE_URI=${REGISTRY_ID}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

if [ -z $(docker image list | awk '{print $1}' | awk -v var="$REP0_NAME" '/$var/') ]; then  
    docker build -f dockerfile -t ${IMAGE_URI} ./   # docker build [PARAMS] PATH
    if [[ $PATH != *"google-cloud-sdk/bin"* ]]; then
        export PATH=~/google-cloud-sdk/bin:$PATH 
    fi
    docker push ${REGISTRY_ID}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAGE}
fi

if [[ "$1"=="run" ]]; then 
    docker run  -it --rm -p 8500:8500  -p 6000:7001  ${REGISTRY_ID}/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAGE}
fi