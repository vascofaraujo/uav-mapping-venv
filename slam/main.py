
import numpy as np

from cv2 import cv2
import itertools
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size
from functions import *

#-----------------------------------------------------------------------------------------


par=import_camera_par()

images_rgb, images_dep=read_images()

get_xyz(images_dep[0], [size(images_dep[0],0), size(images_dep[0],1)],0 ,par.K_depth,1 , 0)

kp1, kp2, mask=match_points(images_rgb[0], images_rgb[1])
cv2.waitKey(1)
#kp1=np.delete(kp1, np.where(kp1== ['<f8','<f8' ]) ,axis=0)
#kp1=np.ma.compress_rowcols(kp1, axis=None)
#kp1=np.delete(kp1, mask)
KP1=[]
for i in range (kp1.shape[0]):
    KP1.append(kp1[i][0])

mask=np.array(mask)

n=np.sum(mask)
#aux=np.zeros((n,2))
aux=[]
for i in range (np.size(mask)):
    if(mask[i]==1):
        aux.append(KP1[i])
img=cv2.drawMarker(images_rgb[0], tuple(aux[0]), (0,0,255))

plt.imshow(img, 'gray'),plt.show()

mask2=np.column_stack((mask,mask))
marr = np.ma.MaskedArray(KP1, mask2=~mask2)
x=np.ma.compress_rows(marr)

print(kp1.shape)
print(kp1[0][0])
print("end")


