#!/bin/sh

for file in outputs/Classify_01/*
do
    python3 %extractor.py $file >> temp/percents.txt
done