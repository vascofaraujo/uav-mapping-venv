import freenect
import cv2 as cv
import numpy as np
# import time

def get_video():
    array,_ = freenect.sync_get_video()
    array = cv.cvtColor(array,cv.COLOR_RGB2BGR)
    return array

#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

if __name__ == '__main__':
    i = 0
    while True:
        frame = get_video()
        depth = get_depth()


        # nameImg = str(time.time()) + ".jpg"
        #Save image when "q" is pressed
        if cv.waitKey(30) & 0xFF == ord('q'):
            cv.imwrite("img-" + str(i) + ".png",frame)
            cv.imwrite("depth-" + str(i) + ".png",depth)
            i += 1
            print("Captured")
            cv.waitKey(0)

        #Show webcam
        cv.imshow("img",frame)
        cv.imshow("d",depth)
        #Increment i for name of image to be saved
        #i = i + 1

        #break on ESC
        if cv.waitKey(30) & 0xFF == 27:
            break
