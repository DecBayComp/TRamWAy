#!/bin/sh

## to get and run the present script, on Maestro:
# cd $MYSCRATCH
# rm -rf TRamWAy; git clone --branch dev https://github.com/DecBayComp/TRamWAy && cd $_
# tests/test-maestro.sh

set -e

version=260217
pyminor=9
container="containers/tramway-hpc-$version-py3$pyminor.sif"
testdata=tests/test_analyzer_200803
slurmopts="-p dbc -q dbc"

[ "`pwd`" = "$MYSCRATCH/TRamWAy" ] || exit 1

# TODO make the data available online again
[ -d "$testdata" ] || cp -r ~/Projects/Tramway/TRamWAy/"$testdata" tests/

module load apptainer
rm -f $container; srun $slurmopts apptainer build $container "containers/tramway-hpc-py3$pyminor"

mkdir -p containers/tmp
srun $slurmopts apptainer exec \
  --bind "$testdata:/root/TRamWAy/$testdata" \
  --cwd=/root/TRamWAy --overlay=containers/tmp $container \
  "python3.$pyminor" -m pytest --ignore=tests/test_analyzer_pipeline.py
