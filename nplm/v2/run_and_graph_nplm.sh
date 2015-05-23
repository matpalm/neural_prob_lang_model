#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <sm|lr> <num_dimensions>"
    exit 1
fi
set -ex

export DTS=`date +%s`
time ./nplm_${1}.py --embedding-dim=$2 --n-hidden=$2 > nplm_${1}.${2}.${DTS}
cat nplm_${1}.${2}.${DTS} | awk -F'	' '{print $1 "\t" $2 "\t" substr($2,0,4) "\t" $3}' > f; mv f nplm_${1}.${2}.${DTS}
#R --vanilla --args $1 $2 $DTS < plot_nplm_XX.R
#echo "graph is nplm_$1.$2.png"