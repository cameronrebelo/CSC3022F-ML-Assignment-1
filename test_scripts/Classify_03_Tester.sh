#!/bin/sh

# Epoch tests
for round in {1..5}
do

    for i in {10..25..5};
    do
        if [ $i -eq 10 ]; then
            python3 src/Classify_01.py $i > outputs/Classify_05/Classify_05_$i\_Epochs_$round.txt
        else
            python3 src/Classify_01.py $i >> outputs/Classify_05/Classify_05_$i\_Epochs_$round.txt
        fi
    done
done