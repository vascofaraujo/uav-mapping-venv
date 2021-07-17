import cv2 as cv
import os
import numpy as np


def read_rgbd():
    images_path = os.getcwd() + "/newpiv/"
    rgbd_array = []

    for i in range(3,10):
        img = cv.imread(images_path+ "rgb_image_" + str(i) + ".png")
        depth = cv.imread(images_path+ "depth_" + str(i) + ".png")

        rgbd = np.zeros((480,640,4), np.uint8)
        rgbd[:,:,0:3] = img
        rgbd[:,:,3] = cv.cvtColor(depth, cv.COLOR_BGR2GRAY)
        rgbd_array.append(rgbd)

    return rgbd_array

def compute_homographies(images):

    image = images[0]
    for i in range(len(images)-1):
        next_image = images[i+1]

        homography_matrix = find_homography_matrix(image, next_image)

        if(homography_matrix is None):
            continue

        stitched_image = compute_warp(image, next_image, homography_matrix)

        stitched_image = remove_black(stitched_image)


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
         if m.distance < 0.3*n.distance:
             good_matches.append(m)
             good_points.append((m.trainIdx, m.queryIdx))


     if len(good_matches) > 8:

         img1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
         img2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])

         M, mask = cv.findHomography(img2_kp, img1_kp, cv.RANSAC,5.0)
         matchesMask = mask.ravel().tolist()

         h = image.shape[0]
         w = image.shape[1]
         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
         dst = cv.perspectiveTransform(pts,M)

         img2 = cv.polylines(gray2,[np.int32(dst)],True,255,3, cv.LINE_AA)


         draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)

         image_matches = cv.drawMatches(image,kp1,next_image,kp2,good_matches,None,**draw_params)
         return M
     return None

def compute_warp(img1, img2, H):

    out_width = img1.shape[1]+img2.shape[1]
    out_height = img1.shape[0]

    output_image = np.zeros((out_height, out_width,3))

    output_image = cv.warpPerspective(img2, H, (out_width, out_height))

    output_image[0:img1.shape[0], 0:img1.shape[1], :] =img1

    return output_image

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
    images = read_images()

    panorama = compute_homographies(images)

    cv.imshow("panorama", panorama)
    while(1):
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyWindow("janela")
