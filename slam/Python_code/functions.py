import os
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy.matlib
from PIL import Image
import struct
import time

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0


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
    images_path = os.getcwd()
    
    cap_rgb=cv2.VideoCapture("rgbd_dataset_freiburg2_pioneer_slam-rgb.avi")
    cap_dep=cv2.VideoCapture("rgbd_dataset_freiburg2_pioneer_slam-depth.avi")
    
    success, image1_rgb=cap_rgb.read()
    success, image2_rgb=cap_rgb.read()
    
    return image1_rgb, image2_rgb

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
        #h,w,d = img1.shape
        #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #dst = cv2.perspectiveTransform(pts,M)
        #images[1] = cv2.polylines(images[1],[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    #draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #               singlePointColor = None,
    #               matchesMask = matchesMask, # draw only inliers
    #               flags = 2)
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    #plt.imshow(img3, 'gray'),plt.show()

    return(src_pts, dst_pts, matchesMask)


def get_pointcloud(color_image,depth_image,camera_intrinsics):
    """ creates 3D point cloud of rgb images by taking depth information
        input : color image: numpy array[h,w,c], dtype= uint8
                depth image: numpy array[h,w] values of all channels will be same
        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image[:,:,0]/camera_intrinsics[0,0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image[:,:,0]/camera_intrinsics[1,1])
    camera_points_z = depth_image[:,:,0]
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image[:,:,0].flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points,color_points

def generate_pointcloud(rgb_file,depth_file,downsample):
    """
    Generate a colored point cloud 
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    transform -- camera pose, specified as a 4x4 homogeneous matrix
    downsample -- downsample point cloud in x/y direction
    pcd -- true: output in (binary) PCD format
           false: output in (text) PLY format
           
    Output:
    list of colored points (either in binary or text format, see pcd flag)
    """
    
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)
    
    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")


    points = []
    point_cloud=[]  
    for v in range(0,rgb.size[1],downsample):
        for u in range(0,rgb.size[0],downsample):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
            if Z==0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
            point_cloud.append([X, Y, Z,color[0],color[1],color[2]])
    file = open('ply_file',"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

    return np.array(point_cloud)


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    if np.linalg.matrix_rank(H) < 3:
        return [], [], 0
        #raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t, 1