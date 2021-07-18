import cv2 as cv
import os
import numpy as np
import open3d as o3d


def read_rgbd():
    images_path = os.getcwd() + "/images/newpiv/"
    rgbd_array = []

    for i in range(3,10):
        img = o3d.io.read_image(images_path+ "rgb_image_" + str(i) + ".png")
        img = np.asarray(img)
        depth = o3d.io.read_image(images_path+ "depth_" + str(i) + ".png")
        depth = np.asarray(depth)
        depth  = np.uint8(depth*(255/depth.max()))

        # img[np.where(depth==0)] = 0

        rgbd = np.zeros((img.shape[0],img.shape[1],4), np.uint8)
        rgbd[:,:,0:3] = img
        rgbd[:,:,3] = depth
        rgbd_array.append(rgbd)

    return rgbd_array

def compute_homographies(images):
    output = []

    image = images[0]
    for i in range(len(images)-1):
        next_image = images[i+1]

        homography_matrix = find_homography_matrix(image, next_image)

        if(homography_matrix is None):
            continue

        stitched_image = compute_warp(image, next_image, homography_matrix)

        stitched_image = remove_black(stitched_image)

        output.append(stitched_image)


        image = stitched_image

    panorama = image

    return panorama

def find_homography_matrix(image, next_image):
     gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
     gray2 = cv.cvtColor(next_image, cv.COLOR_BGR2GRAY)

     sift = cv.SIFT_create()

     kp1, des1 = sift.detectAndCompute(gray, None)
     kp2, des2 = sift.detectAndCompute(gray2, None)

     #flann matches
     FLANN_INDEX_KDTREE=0
     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
     search_params = dict(checks = 50)
     flann = cv.FlannBasedMatcher(index_params, search_params)

     matches = flann.knnMatch(des1, des2, k=2)

     #get rid of bad matches
     good_matches = []
     good_points = []

     for m,n in matches:
         if m.distance < 0.6*n.distance:
             good_matches.append(m)
             good_points.append((m.trainIdx, m.queryIdx))


     if len(good_matches) > 8:

         img1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
         img2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])

         M, _ = cv.findHomography(img2_kp, img1_kp, cv.RANSAC,5.0)

         return M
     return None

def compute_warp(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min,-y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img
    # out_width = img1.shape[1]+img2.shape[1]
    # out_height = img1.shape[0]#+img2.shape[0]
    #
    # output_image = np.zeros((out_height, out_width,3))
    #
    # output_image = cv.warpPerspective(img2, H, (out_width, out_height))
    #
    # #output_image[0:img1.shape[0], 0:img1.shape[1], :] = img1
    #
    # return output_image

def remove_black(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    pixel_sum = np.sum(gray, axis=0).tolist()
    first = True

    for index, value in enumerate(pixel_sum):
        if value == 0:
            continue
        else:
            ROI = image[0:image.shape[0], index:index+1]
            if first:
                result = image[0:image.shape[0], index+1:index+2]
                first= False
                continue
            result = np.concatenate((result, ROI), axis=1)
    return result


if __name__ == '__main__':
    images = read_rgbd()

    panorama = compute_homographies(images)

    cv.imshow("panorama", panorama)
    while(1):
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyWindow("janela")
