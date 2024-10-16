#!/usr/bin/env bash

mkdir ${PWD}/tmp

docker run --rm \
  -v ${PWD}:/local openapitools/openapi-generator-cli generate \
  -i /local/agentsmith-agent/resources/openai.yaml \
  -g rust \
  -o /local/tmp

sudo chmod -R 777 ${PWD}/tmp
