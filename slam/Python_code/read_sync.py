# file to read from sync 

import sys
import os
import numpy

def read_sync(filename):
    """


    """

    file = open(filename)
    data = file.read()
    lines=data.split("\n")
    del data
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] \
        for line in lines if len(line)>0 and line[0]!="#"]
    for i in range(len(list)):
        for j in range(2):
            list[i][j]=list[i][j] + '.png'
    return list
