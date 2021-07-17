import cv2 as cv
import os
import numpy as np
import itertools


#READ THIS
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
def read_images():
    images_path = os.getcwd + "/casa2/"

    images = []

    scale_percent = 20

    for i in range(1,8):
        print(i)
        img = cv.imread(images_path+ str(i) + ".jpg")
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        images.append(cv.resize(img, dim))

    return images


def stitch_images(images):

    images_stitches = []
    panoramas = []

    for i, image in enumerate(images):
        print(i)
        if i == len(images)-1:
            continue

        next_image = images[i+1]

        out_image, panorama = compute_stitches(image, next_image)
        if out_image is not None:
            images_stitches.append(out_image)
            panoramas.append(panorama)

    return images_stitches, panoramas

def compute_stitches(image, next_image):
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

     if len(good_matches) > 10:

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


         panorama = compute_warp(image, next_image, M)


         return image_matches, panorama
     return None, None

def compute_warp(img1, img2, H):

    out_width = img1.shape[1]+img2.shape[1]
    out_height = img1.shape[0]

    output_image = np.zeros((out_height, out_width,3))

    output_image = cv.warpPerspective(img2, H, (out_width, out_height))

    output_image[0:img1.shape[0], 0:img1.shape[1], :] =img1

    return output_image

if __name__ == '__main__':
    images = read_images()

    images_stitches, panoramas = stitch_images(images)

    for panorama, stitch in zip(panoramas, images_stitches):
    #for panorama in panoramas:
        cv.imshow("panorama", panorama)
        cv.imshow("stitch", stitch)

        while(1):
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyWindow("janela")
