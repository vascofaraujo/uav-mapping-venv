import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from panorama_functions import *

class PointCloud():
    def __init__(self):

        self.img = o3d.io.read_image("panorama-rgb.png")
        self.depth = o3d.io.read_image("panorama-depth.png")
        # self.img = o3d.io.read_image("img-1625963495.1012716.jpg")
        # self.depth = o3d.io.read_image("depth-1625963495.1012716.jpg")

        self.camera_matrix = np.loadtxt('camera-matrix-kinect.txt', delimiter =',')
        self.camera_distortion = np.loadtxt('camera-distortion-kinect.txt', delimiter =',')

        self.width = 640
        self.height = 480
        self.fx = self.camera_matrix[0][0]
        self.fy = self.camera_matrix[1][1]
        self.cx = self.camera_matrix[0][2]
        self.cy = self.camera_matrix[1][2]

    def create_pcd(self):

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.img, self.depth)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, o3d.camera.PinholeCameraIntrinsic(
                self.width, self.height, self.fx, self.fy, self.cx, self.cy))
        return pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    def create_rgbd(self):
        rgbd = np.zeros((self.height,self.width,4), np.uint8)
        rgbd[:,:,0:3] = self.img
        rgbd[:,:,3] = self.depth


        cv.imshow("a", rgbd)
        cv.waitKey(0)


if __name__ == "__main__":
    images = read_rgbd()
    panorama = compute_homographies(images)


    cv.imshow("panorama", panorama)

    while(1):
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyWindow("panorama")

    cv.imwrite("panorama-rgb.png", panorama[:,:,0:3])
    cv.imwrite("panorama-depth.png", panorama[:,:,3])

    pc = PointCloud()
    #pc.create_rgbd()

    point_cloud = pc.create_pcd()

    o3d.visualization.draw_geometries([point_cloud])
