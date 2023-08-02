#!/bin/bash

gpus=1
name="Magnitudes_Test"
# epochs=15000

for epochs in 1500 3000; do
    python thisquakedoesnotexist/condensed_code/combo.py --gpu $gpus --name $name --epochs $epochs
done
