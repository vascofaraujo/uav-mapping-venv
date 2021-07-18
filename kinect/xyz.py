import numpy as np
import cv2 as cv
import open3d as o3d
from panorama_functions import *
import sys


class PointCloud():
    def __init__(self):
        self.camera_matrix = np.loadtxt('camera-matrix-kinect.txt', delimiter =',')
        self.fx = self.camera_matrix[0][0]
        self.fy = self.camera_matrix[1][1]
        self.cx = self.camera_matrix[0][2]
        self.cy = self.camera_matrix[1][2]

    def get_images(self, rgb="images/newpiv/rgb_image_8.png", depth="images/newpiv/depth_8.png"):
        img = o3d.io.read_image(rgb)
        self.img = np.asarray(img)

        depth = o3d.io.read_image(depth)
        depth = np.asarray(depth)
        self.depth = np.uint8(depth*(255/depth.max()))
        # self.depth[self.depth >= 0] = 25
        # self.depth[self.depth >= 50] = 75
        # self.depth[self.depth >= 100] = 125
        # self.depth[ self.depth >= 150] = 175
        # self.depth[self.depth >= 200] = 225


        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

    def get_xyz(self):
        xyz = np.zeros((self.height * self.width, 3))
        color = np.zeros((self.height * self.width, 3))

        i = 0
        for v in range(self.height):
            for u in range(self.width):
                if self.depth[v,u] < 150:
                    continue
                x = (u - self.cx) * self.depth[v, u] / self.fx
                y = (v - self.cy) * self.depth[v, u] / self.fy
                z = self.depth[v, u]
                xyz[i] = (x, y, z)
                color[i] = self.img[v,u]/255
                i += 1

        xyz = xyz[0:i]
        color = color[0:i]

        return xyz, color


    def visualize_pcd(self, xyz, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(color)

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        o3d.io.write_point_cloud("pcd2.ply", pcd)

        pcd_load = o3d.io.read_point_cloud("pcd2.ply")
        o3d.visualization.draw_geometries([pcd_load])


if __name__ == "__main__":

    pc = PointCloud()

    arg = sys.argv[1:]
    if (arg):
        images = read_rgbd()

        panorama = compute_homographies(images)

        #cv.destroyWindow("q")

        cv.imwrite("panorama-rgb.png", panorama[:,:,0:3])
        cv.imwrite("panorama-depth.png", panorama[:,:,3])

        pc.get_images("panorama-rgb.png", "panorama-depth.png")
    else:
        pc.get_images()

    cv.imshow("q", pc.depth)
    cv.waitKey(0)
    xyz, color = pc.get_xyz()

    pc.visualize_pcd(xyz, color)
