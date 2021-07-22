import sys
import os
import numpy as np
from matplotlib import pyplot as plt

path=os.getcwd() + "/rgbd_dataset_freiburg2_pioneer_slam/"

filename=path + "groundtruth.txt"

file = open(filename)
data = file.read()
lines=data.split("\n")
del data
list = [[v.strip() for v in line.split(" ") if v.strip()!=""] \
    for line in lines if len(line)>0 and line[0]!="#"]

y=np.array(list)
x=[]

downsample=100
for i in range(1, len(y), downsample):
    x.append([float(y[i,1]),float( y[i,2]), float(y[i,3])])
x=np.array(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(x[:,0], x[:,1], x[:,2], marker = 'x')
plt.show()

print("end")