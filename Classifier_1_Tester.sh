#!/bin/sh

#Epoch tests
# for i in {10..40..5};
# do
#     if [ $i -eq 10 ]; then
#         python3 src/Classify_01.py $i > outputs/Classify_01_Epoch_Test.txt
#     else
#         python3 src/Classify_01.py $i >> outputs/Classify_01_Epoch_Test.txt
#     fi
# done
#Learning rate
python3 src/Classify_01.py 0.1 > outputs/Classify_01_LR_Test.txt
python3 src/Classify_01.py 0.01 >> outputs/Classify_01_LR_Test.txt
python3 src/Classify_01.py 0.001 >> outputs/Classify_01_LR_Test.txt
python3 src/Classify_01.py 0.0001 >> outputs/Classify_01_LR_Test.txt
python3 src/Classify_01.py 0.00001 >> outputs/Classify_01_LR_Test.txt