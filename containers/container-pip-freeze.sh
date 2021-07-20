#!/bin/bash

if [ -z $1 ]; then
  echo "usage: $0 CONTAINER_FILE [PATH_TO_SINGULARITY]"
  exit 1
fi

CONTAINER=$1
shift

if [ -z $1 ]; then
  SINGULARITY="singularity"
else
  SINGULARITY=$1
fi

$SINGULARITY exec "$CONTAINER" python3.6 -m pip freeze

