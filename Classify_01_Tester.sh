#!/bin/sh

#Activation Function Tests

actiation_function=" "
for round in {4..5};
do
    for mode in {1..4};
    do
        for i in {5..20..5};
        do
            if [ $mode -eq 1 ]; then
                actiation_function=Tanh
            fi
            if [ $mode -eq 2 ]; then
                actiation_function="Sigmoid"
            fi
            if [ $mode -eq 3 ]; then
                actiation_function="ReLU"
            fi
            if [ $mode -eq 4 ]; then
                actiation_function="LeakyReLU"
            fi
            if [ $i -eq 5 ]; then
                python3 src/Classify_03.py $i $mode > outputs/Classify_01/Classify_01_$actiation_function\_$round.txt
            else
                python3 src/Classify_03.py $i $mode >> outputs/Classify_01/Classify_01_$actiation_function\_$round.txt
            fi
        done
    done
done