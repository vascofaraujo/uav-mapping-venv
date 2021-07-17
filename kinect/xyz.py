import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d

class PointCloud():
    def __init__(self):
        self.img = cv.imread("rgb_image_1.png")
        depth = cv.imread("depth_1.png")
        depth = np.uint8(depth*(255/depth.max()))
        self.depth = cv.cvtColor(depth,cv.COLOR_BGR2GRAY)

        self.camera_matrix = np.loadtxt('camera-matrix-kinect.txt', delimiter =',')
        self.camera_distortion = np.loadtxt('camera-distortion-kinect.txt', delimiter =',')

        self.width = 640
        self.height = 480
        self.fx = self.camera_matrix[0][0]
        self.fy = self.camera_matrix[1][1]
        self.cx = self.camera_matrix[0][2]
        self.cy = self.camera_matrix[1][2]
        self.scaling_factor = 0.001


    def get_xyz(self):
        u = np.tile(np.arange(self.width), self.height)
        u = u[:] - self.cx
        v = np.tile(np.transpose(np.arange(self.height)), self.width)
        v = v[:] - self.cy;

        dep_vec = np.reshape(self.depth, -1)

        xyz = np.zeros((self.height * self.width, 3))

        xyz[:,0] = u * dep_vec / self.fx
        xyz[:,1] = v * dep_vec / self.fy
        xyz[:,2] = dep_vec

        xyz = np.zeros((self.height * self.width, 3))
        i = 0
        for v in range(self.height):
            for u in range(self.width):
                x = (u - self.cx) * self.depth[v, u] / self.fx
                y = (v - self.cy) * self.depth[v, u] / self.fy
                z = self.depth[v, u]
                xyz[i] = (x, y, z)
                i += 1

        color = np.reshape(self.img, [self.height * self.width, 3]);
        color = color/255;

        return xyz, color


    def visualize_pcd(self, xyz, color):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #
        # o3d.io.write_point_cloud("pcd.ply", pcd)
        #
        # pcd_load = o3d.io.read_point_cloud("pcd.ply")
        # o3d.visualization.draw_geometries([pcd_load])

        ax = plt.axes(projection='3d')
        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = color, s=0.01)
        plt.show()

if __name__ == "__main__":
    pc = PointCloud()

    xyz, color = pc.get_xyz()
    pc.visualize_pcd(xyz, color)
