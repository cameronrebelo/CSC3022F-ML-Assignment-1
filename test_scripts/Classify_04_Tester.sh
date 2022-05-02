#!/bin/sh

#Learning rate

for round in {1..5}
do
    python3 src/Classify_01.py 0.1 > outputs/Classify_06/Classify_06_0.1_LR_$round.txt
    python3 src/Classify_01.py 0.01 >> outputs/Classify_06/Classify_06_0.01_LR_$round.txt
    python3 src/Classify_01.py 0.001 >> outputs/Classify_06/Classify_06_0.001_LR_$round.txt
    python3 src/Classify_01.py 0.0001 >> outputs/Classify_06/Classify_06_0.0001_LR_$round.txt
done