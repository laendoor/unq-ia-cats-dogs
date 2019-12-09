#!/bin/bash

DATAPATH='train_1500'
N=1500
PADDING='edge'
EPOCHS=(50 100 200 300 500)

echo "*** Script configuration ***"
echo "PATH    = $DATAPATH"
echo "DATASET = $N"
echo "EPOCHS  = ${EPOCHS[*]}"
echo ""

for e in "${EPOCHS[@]}"; do
  FILE="dataset.${N}_epoch.${e}_padding.${PADDING}.md"
  COMMAND="python gato.py --epoch=$e --padding=$PADDING --path=$DATAPATH"

  echo ""
  echo "Running: ${COMMAND}"
  eval "$COMMAND | tee output/$FILE"
done
