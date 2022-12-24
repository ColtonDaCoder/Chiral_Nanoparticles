#!/bin/bash
TIC=`date +%s.%N`
CASE_DIR=$2
NTHREADS=$1
source ~/.bash_profile
cd /work/coltonb/gpar/$CASE_DIR
python ./run_sim.py -t $NTHREADS
TOC=`date +%s.%N`
J1_TIME=`echo "$TOC - $TIC" | bc -l`
echo "This run took=$J1_TIME sec using $CASE_DIR with $NTHREADS threads on $HOSTNAME"
