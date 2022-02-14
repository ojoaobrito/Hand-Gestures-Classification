import cv2
import numpy as np
import sys

def callback(value):
    pass

def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)

    for i in ["MIN", "MAX"]:
        v = 0 if(i=="MIN") else 255

        for j in range_filter:
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return(values)

def main():

    range_filter = "HSV"

    fourcc = cv2.VideoWriter_fourcc("m","p","4","v")
    
    with open("names_work.txt") as file:
        videos = list(map(lambda x : x.split("\n")[0],file.readlines()))

    for i in videos:
        print("\n" + i + "\n")
        saved_frames = []

        out_writer = cv2.VideoWriter("out/" + i[7:], fourcc, 30.0, (640,640))
        v1_min = 0
        v2_min = 0
        v3_min = 0
        v1_max = 0
        v2_max = 0
        v3_max = 0

        camera = cv2.VideoCapture(i)

        setup_trackbars(range_filter)

        while True:
            ret, image = camera.read()

            if(not ret): # reset video

                user = input("Save video? ")

                if(user==""): # save the video
                    print("lower: [" + str(v1_min) + ", " + str(v2_min) + ", " + str(v3_min) + "]")
                    print("upper: [" + str(v1_max) + ", " + str(v2_max) + ", " + str(v3_max) + "]")

                    for i in saved_frames:
                        out_writer.write(i)

                    out_writer.release()
                    break

                else:
                    camera = cv2.VideoCapture(i)
                    saved_frames = []
                    v1_min = 0
                    v2_min = 0
                    v3_min = 0
                    v1_max = 0
                    v2_max = 0
                    v3_max = 0
                    continue

            image = cv2.resize(image,(640,640),interpolation=cv2.INTER_AREA)

            if(range_filter=="RGB"):
                frame_to_thresh = image.copy()
            else:
                frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(range_filter)

            thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            thresh = cv2.erode(thresh, kernel, iterations = 1)
            thresh = cv2.dilate(thresh, kernel, iterations = 1)

            thresh = cv2.GaussianBlur(thresh, (3, 3), 0)

            skin = cv2.bitwise_and(image, image, mask = thresh)
            
            cv2.imshow("images", np.hstack([image, skin]))

            saved_frames.append(skin)

            if(cv2.waitKey(1) & 0xFF is ord('q')): # quit the program
                sys.exit()
                cv2.destroyAllWindows()
            
            cv2.waitKey(65)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()