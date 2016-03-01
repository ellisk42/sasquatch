#!/bin/bash
BIN_DIR=/afs/csail.mit.edu/@sys/local/bin
AUTHLOOP=$BIN_DIR/authloop
AUTHOUT=/tmp/$USER.auth.$$


#kinit -r 8d
#aklog
#nohup $AUTHLOOP $KEYTAB >> $AUTHOUT &

ITERATIONS=$1
PREFIX=$2
shift
shift
#COMMAND=$3

for j in `seq 1 $ITERATIONS`;
do
    longjob -o $PREFIX$j "$@"
#    nohup $COMMAND >> $PREFIX$j 2>&1 &
done

