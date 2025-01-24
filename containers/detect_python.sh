#!/bin/bash

for ((minor=13;6<=minor;minor--)); do
py=python3.$minor
if command -v $py &>/dev/null; then
if $py -m pip freeze | grep tramway &>/dev/null; then
echo $py
exit 0
fi
fi
done

py=python2.7
if command -v $py &>/dev/null; then
if [ -z "$($py -m pip show -q tramway 2>&1)" ]; then
echo $py
exit 0
fi
fi

exit 1
