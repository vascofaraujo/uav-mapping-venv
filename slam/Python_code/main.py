import numpy as np

from cv2 import cv2
import itertools
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from functions import *
from PIL import Image
from read_sync import read_sync
import time

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0

#-----------------------------------------------------------------------------------------

fx = 525.0  # focal length x
fy = 525.0  # focal length y
cx = 319.5  # optical center x
cy = 239.5  # optical center y

cam_int  = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


images_path=os.getcwd() + "/rgbd_dataset_freiburg2_pioneer_slam/"

#read from .txt

frames=read_sync('sync.txt')


x=np.array([[0],[0],[0]])
y=np.transpose(x)
for j in range(400):
    if j==0:
        image1_rgb=cv2.imread(images_path+ "rgb/" + frames[j][0])
        image2_rgb=cv2.imread(images_path+ "rgb/" + frames[j+1][0])

        depth1 = Image.open(images_path+ "depth/" + frames[j][1])
        depth2 = Image.open(images_path+ "depth/" + frames[j+1][1])
    else:
        image1_rgb=image2_rgb
        depth1=depth2
        image2_rgb=cv2.imread(images_path+ "rgb/" + frames[j+1][0])
        depth2 = Image.open(images_path+ "depth/" + frames[j+1][1])

    #print(time.time() - TIME)

    #downsample=10
    #points=generate_pointcloud(images_path+ "rgb/1311878194.274451.png",images_path+ "depth/1311878194.329554.png",downsample)

    TIME=time.time()
    kp1, kp2, mask=match_points(image1_rgb, image2_rgb)
    print(time.time() - TIME)

    
    #A=np.array([[0],[0],[0]])
    #B=np.array([[0],[0],[0]])
    A=[]
    B=[]
    for i in range(len(mask)):
        if mask[i]==1:
            Z1 = depth1.getpixel((np.round(kp1[i,0][0]),np.round(kp1[i,0][1]))) / scalingFactor
            Z2 = depth2.getpixel((np.round(kp2[i,0][0]),np.round(kp2[i,0][1]))) / scalingFactor
            if (Z1==0) or (Z2==0): continue
            X1 = (kp1[i,0][0] - centerX) * Z1 / focalLength
            Y1 = (kp1[i,0][1] - centerY) * Z1 / focalLength
            X2 = (kp2[i,0][0] - centerX) * Z2 / focalLength
            Y2 = (kp2[i,0] [1]- centerY) * Z2 / focalLength
            A.append([X1,Y1,Z1])
            B.append([X2,Y2,Z2])
            #A=np.column_stack((A, [X1,Y1,Z1]))
            #B=np.column_stack((B, [X2,Y2,Z2]))
    A=np.transpose(np.array(A))
    B=np.transpose(np.array(B))
    
    R,t, suc=rigid_transform_3D(A, B)
    if suc==1:
        x=np.matmul(R,x) + t
        if j %5==0:
            y=np.append(y,np.transpose(x), axis=0)
    #print(x)
    #print(time.time() - TIME)



#create the graph
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot(y[:,0], y[:,1], y[:,2], marker = 'x')
plt.show()

print("end")


