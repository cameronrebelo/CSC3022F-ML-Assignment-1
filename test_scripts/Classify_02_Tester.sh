#!/bin/sh

#Optimiser Tests

optimizer=" "
for round in {1..3};
do
    for mode in {1..2};
    do
        if [ $mode -eq 1 ]; then
            optimizer="SGD"
        fi
        if [ $mode -eq 2 ]; then
            optimizer="Adam"
        fi
        python3 src/Classify_02.py $mode > outputs/Classify_02/Classify_02_$optimizer\_$round.txt
    done
done