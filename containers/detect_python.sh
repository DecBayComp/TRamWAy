#!/bin/bash

for ((minor=12;6<=minor;minor--)); do
py=python3.$minor
if [ -x "$(command -v $py)" ]; then
if [ -z "$($py -m pip show -q tramway 2>&1)" ]; then
echo $py
exit 0
fi
fi
done

py=python2.7
if [ -x "$(command -v $py)" ]; then
if [ -z "$($py -m pip show -q tramway 2>&1)" ]; then
echo $py
exit 0
fi
fi

exit 1

