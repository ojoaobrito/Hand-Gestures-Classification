import os, sys, csv
import cv2
import numpy as np
import argparse
from shutil import copyfile, rmtree
from random import randint, uniform, choice

def create_folders(): # auxiliary function, creates every necessary folder

    if(os.path.exists("hand_gestures_dataset")):rmtree("hand_gestures_dataset")

    os.makedirs("hand_gestures_dataset")
    os.makedirs("hand_gestures_dataset/train")
    os.makedirs("hand_gestures_dataset/test")

def rename_video(video): # auxiliary function, renames the given video to the correct format

    video_name = video.split(".")[0]
    lighting = {"0": ["morning_qmae","morning_qmano","morning_qpai","night_pai","night_mae","night_mano"], "1": ["afternoon_mae","afternoon_mano","afternoon_pai","afternoon_qpai","afternoon_qmae","afternoon_qmano","afternoon_zmae","afternoon_zmano","afternoon_zpai","morning_mae","morning_mano","morning_pai","morning_zmae","morning_zmano","morning_zpai","night_qmae","night_qmano","night_qpai","night_zmano","night_zmae","night_zpai"]}
    background = {"0": ["afternoon_qmae","afternoon_qmano","afternoon_qpai","morning_qmae","morning_qpai","morning_qmano","night_qmae","night_qmano","night_qpai"], "1": ["afternoon_mae","afternoon_pai","afternoon_mano","morning_pai","morning_mae","morning_mano","night_mae","night_pai","night_mano"], "2": ["afternoon_zmae","afternoon_zpai","afternoon_zmano","morning_zmae","morning_zmano","morning_zpai","night_zmae","night_zpai","night_zmano"]}
    distance = {"0": ["afternoon_mae","afternoon_pai","afternoon_mano","afternoon_zmae","afternoon_zpai","afternoon_zmano","morning_mae","morning_pai","morning_mano","morning_qmae","morning_qpai","morning_qmano","morning_zmae","morning_zmano","morning_zpai","night_mae","night_pai","night_mano","night_zmae","night_zpai","night_zmano"], "1": ["afternoon_qpai","afternoon_qmano","afternoon_qmae","night_qmae","night_qpai","night_qmano"]}

    # --------------------------------------------------
    # class
    # --------------------------------------------------
    class_number = video_name[-1]
    new_name = class_number + "_" + video_name[-1] + "_"

    # ------------------------------------------
    # subject
    # ------------------------------------------
    if("mae" in video_name): new_name += "0_"
    elif("mano" in video_name): new_name += "1_"
    elif("pai" in video_name): new_name += "2_"
    else: new_name += "4_"

    # ------------------------------------
    # lighting
    # ------------------------------------
    if("test" in video):
        if("z" in video): new_name += "1_"
        else: new_name += "0_"

    else:
        for k,v in lighting.items():
            for i in v:
                if(i in video):
                    new_name += k + "_"
                    break
    
    # -----------------------------------
    # background
    # -----------------------------------
    if("test" in video): new_name += "0_"

    else:
        for k,v in background.items():
            for i in v:
                if(i in video):
                    new_name += k + "_"
                    break
    
    # ------------------------------------
    # distance
    # ------------------------------------
    if("test" in video):
        if("z" in video): new_name += "0_"
        else: new_name += "1_"

    else:
        for k,v in distance.items():
            for i in v:
                if(i in video):
                    new_name += k + "_"
                    break

    # rename the video
    if("test" in video): copyfile(video,"hand_gestures_dataset/test/" + new_name[:-1] + ".m4v")
    else: copyfile(video,"hand_gestures_dataset/train/" + new_name[:-1] + ".m4v")

def parse_frames(current): # auxiliary function, parses every video in "NUM_FRAMES" frames

    if(os.path.isdir(current)): # we've reached a directory, let's explore it
        for i in os.listdir(current):
            if(i[0]!="."):
                parse_frames(current + "/" + i)

    else: # we've reached a video file, let's parse it into "NUM_FRAMES" frames

        print(current)

        vidcap = cv2.VideoCapture(current)
        
        # calculate the total number of frames of the input video
        try:
            num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            pass
        
        # get equidistant frames
        frames = np.linspace(0,(num_frames-1),NUM_FRAMES).astype(int)

        # save the frames
        counter = 0
        while(counter<num_frames):
            success,image = vidcap.read()
            if(counter in frames and success): 
                
                # -----------------------------------------------
                # (possibly) flip the image
                # -----------------------------------------------
                chance = uniform(0,1)

                if(chance<=FLIP_PROB): image = cv2.flip(image, 1)

                # -----------------------------------------------------------------
                # (possibly) rotate the image
                # -----------------------------------------------------------------
                chance = uniform(0,1)

                if(chance<=ROTATE_PROB):

                    angle = randint(-3,3)

                    def rotate_image(image, angle):
                        h, w = image.shape[:2]
                        M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
                        image = cv2.warpAffine(image, M, (w, h))
                        return image

                    def crop_image(image, angle):
                        h = image.shape[0]
                        w = image.shape[1]
                        tan_a = abs(np.tan(angle * np.pi / 180))
                        b = int(tan_a / (1 - tan_a ** 2) * (h - w * tan_a))
                        d = int(tan_a / (1 - tan_a ** 2) * (w - h * tan_a))
                        return image[d:h - d, b:w - b]

                    image = rotate_image(image, angle)

                    # crop the image to remove the black corners
                    image = crop_image(image,angle)

                # -------------------------------------------------------------------------------------------------
                # (possibly) change the image's contrast
                # -------------------------------------------------------------------------------------------------
                chance = uniform(0,1)

                if(chance<=CONTRAST_PROB): 
                    ALPHA = choice([0.85,0.95,1.05,1.15,1.25,1.35,1.45]) # controls the contrast levels (1.0 - 3.0)
                    BETA = 0 # controls the brightness levels (0 - 100)
                    
                    image = cv2.convertScaleAbs(image, alpha=ALPHA, beta=BETA)

                # save the final image
                cv2.imwrite(current.replace(".m4v","f%d.jpg") % counter, image)

                # convert to black and white
                image = cv2.imread(current.replace(".m4v","f%d.jpg") % counter)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(current.replace(".m4v","f%d.jpg") % counter, image)

            counter += 1

        os.remove(current)

if __name__ == "__main__":

    ##########################################################################################################################
    # PARSE THE ARGUMENTS
    ##########################################################################################################################
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument("--num_frames", required=False, help="Amount of frames withdrawn from each video", default=25)
    args_parser.add_argument("--flip_prob", required=False, help="Probability of flipping images", default=0.6)
    args_parser.add_argument("--rotate_prob", required=False, help="Probability of rotating images", default=0.85)
    args_parser.add_argument("--contrast_prob", required=False, help="Probability of changing images' contrast", default=0.90)

    args = vars(args_parser.parse_args())

    # ----------------------------------------------
    # initialize global variables with the arguments
    # ----------------------------------------------
    NUM_FRAMES = args["num_frames"]
    FLIP_PROB = args["flip_prob"]
    ROTATE_PROB = args["rotate_prob"]
    CONTRAST_PROB = args["contrast_prob"]

    #######################################
    # SETUP
    #######################################
    create_folders()

    # get the source videos
    source_videos = os.listdir("segmented")

    ######################################
    # RENAME THE VIDEO FILES
    ######################################
    print("RENAME VIDEOS")
    for i in source_videos:
        if(i[0]!="."):
            rename_video("segmented/" + i)

    ################################################
    # PARSE THE VIDEO FILES INTO "NUM_FRAMES" FRAMES
    ################################################
    print("PARSE VIDEOS")
    parse_frames("hand_gestures_dataset")