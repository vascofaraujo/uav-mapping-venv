import cv2 as cv
from panorama_functions import *


if __name__ == "__main__":
    images = read_rgbd()
    # img = []
    # for i in images:
    #     img.append(i[:,:,0:3])

    stitcher = cv.Stitcher_create()
    _, result = stitcher.stitch((images[0], images[1]))
    a= stitcher.estimateTransform((images[0], images[1]))
    print(a)


    cv.imshow("l",result)
    cv.waitKey(0)
    cv.destroyAllWindows()
