#!/bin/bash

if [ -z $1 ]; then
  echo "usage: $0 CONTAINER_FILE [PATH_TO_SINGULARITY]"
  exit 1
fi

CONTAINER=$1
shift

if [ -z $1 ]; then
  SINGULARITY="singularity exec"
else
  SINGULARITY="$1 exec"
fi

if [ -d /pasteur ]; then
  SINGULARITY="$SINGULARITY -B /pasteur"
fi

PYTHON=$($SINGULARITY "$CONTAINER" ./detect_python.sh)
echo $PYTHON

$SINGULARITY "$CONTAINER" $PYTHON -s -m pip freeze

