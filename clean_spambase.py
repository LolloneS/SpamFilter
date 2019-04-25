#!/usr/bin/env python3

'''
Deletes lines whose [0, 54] values are all 0's from the 
dataset and writes the "good" lines to spambase_clean.data
'''
with open('spambase.data', 'r') as f:
    with open('spambase_clean.data', 'w') as ff:
        for line in f:
            splitted = [float(k) for k in line.split(',')]
            splitted = splitted[:54] 
            if sum(splitted) != 0:
                ff.write(line)
