#!/bin/sh

# Epoch tests
for round in {1..2}
do

    for i in {15..25..5};
    do
        if [ $i -eq 15 ]; then
            python3 src/Classify_03.py $i > outputs/Classify_03/Classify_03_$i\_Epochs_$round.txt
        else
            python3 src/Classify_03.py $i >> outputs/Classify_03/Classify_03_$i\_Epochs_$round.txt
        fi
    done
done