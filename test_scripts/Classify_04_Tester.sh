#!/bin/sh

#Learning rate

for round in {1..3}
do
    python3 src/Classify_04.py 0.1 > outputs/Classify_04/Classify_04_0.1_LR_$round.txt
    python3 src/Classify_04.py 0.01 >> outputs/Classify_04/Classify_04_0.01_LR_$round.txt
    python3 src/Classify_04.py 0.001 >> outputs/Classify_04/Classify_04_0.001_LR_$round.txt
    python3 src/Classify_04.py 0.0001 >> outputs/Classify_04/Classify_04_0.0001_LR_$round.txt
done