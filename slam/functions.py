import os
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy.matlib

class camera:
    def __init__(self, K_rgb, K_depth, R,T):
        self.K_rgb=K_rgb
        self.K_depth=K_depth
        self.T=T
        self.R=R
    def show(self):
        print('K_rgb\n', self.K_rgb)
        print('K_depth\n', self.K_depth)
        print('R\n', self.R)
        print('T\n', self.T)



def import_camera_par():
    path = os.getcwd() + "/camera_par/"
    K_rgb=np.load(path + 'K_rgb.npy')
    K_depth=np.load(path + 'K_depth.npy')
    R=np.load(path + 'R.npy')
    T=np.load(path + 'T.npy')
    par=camera(K_rgb, K_depth, R,T) 
    return par


def read_images():
    images_path = os.getcwd() + "/short/"

    images_rgb = []
    images_dep=[]

    for i in range(12,15):
        images_rgb.append(cv2.imread(images_path+ "rgb_image_" + str(i) + ".png"))
    
    for i in range(1,4):
        images_dep.append(cv2.imread(images_path+ "depthc_" + str(i) + ".png"))

    return images_rgb, images_dep

def match_points(img1, img2):
    MIN_MATCH_COUNT = 10
    sift=cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        #images[1] = cv2.polylines(images[1],[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()

    return(src_pts, dst_pts, matchesMask)

def get_xyz(im_vec, im_orig_size='', good_inds='', K='', alpha='', beta=''):
    #convert the data from the depth image into the 3d reference and then relate it
    # to the one from the rgb 
    
    Kx = K[0,0]
    Cx = K[0,2]
    Ky = K[1,1]
    Cy = K[1,2]

    size1=im_orig_size[0]
    size2=im_orig_size[1]
    
    u=numpy.matlib.repmat(np.arange(size2), size1,1)
    u= u - Cx

    v=numpy.matlib.repmat(np.transpose(np.arange(size1)), size2,1)
    v= v - Cy


    # save z position
    # u = repmat(1:im_size(2),im_size(1),1);
    # u = u(:)-Cx;
    # v = repmat((1:im_size(1))',im_size(2),1);
    # v=v(:)-Cy;
    # xyz(:,1)=xyz(:,3)/Kx .* u;
    # xyz(:,2)=xyz(:,3)/Ky .* v;


    return 0