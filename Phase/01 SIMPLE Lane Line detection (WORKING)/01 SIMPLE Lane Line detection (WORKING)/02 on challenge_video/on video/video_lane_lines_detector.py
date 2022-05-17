from turtle import delay, shape, width
from matplotlib import image        # import image plot type from the lib
from moviepy.editor import VideoFileClip
import matplotlib.pylab as plt      # for plot
import cv2                          # for computer vision
import numpy as np                  # for math



##########################################################################
### define a funcition to mask the region of interest from input image ###

def region_of_interest (input_image, vertices):
    mask = np.zeros_like(input_image)               
    
    # channel_count = input_image.shape[2]            # no. of colors channels in the image
    match_mask_color = 255                            # make it one color has scale 255

    cv2.fillPoly(mask, vertices, match_mask_color)

    after_masking_image = cv2.bitwise_and(input_image, mask)

    return after_masking_image



####################################################
### create a fn to draw lines on the input image ###

#take two parameters (original_image - lines created)
def draw_lines (img, lines):
    img = np.copy(img)

    #create a blank image with the same dimensions of the original image
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), thickness= 3)
    
    # merge original image with the blank image
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img



######################################
### define a fn include everything ###

def process(image):
    ##############################
    ### image height and width ###
    
    height = image.shape[0]         # 720      -> shape[0] = 720
    width = image.shape[1]          # 1200     -> shape[1] = 1200

    ###########################################
    ### define the region of interest edges ###

    region_of_interest_vertices = [
        # (width, height)
        (330, 700),
        (600, 535),
        (700, 495),
        (800, 495),
        (1070, 700)
    ]

    ####################################################################
    ### convert the input image to grayscale then apply canny filter ###

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 200, 230)       # what's canny ?! canny is a filter of converting the image to lines and strokes

    ###########################################################################
    ### call the fn and give it the canny image & interest edges identified ###

    masked_image_for_work = region_of_interest(
        canny_image,
        np.array([region_of_interest_vertices], np.int32,)
    )

    ################################################################################
    ### get lines from the masked_image_for_work by using hough transform method ###

    lines = cv2.HoughLinesP(masked_image_for_work,
                            rho=6,
                            theta=np.pi/60,
                            threshold=100,
                            lines=np.array([]),
                            minLineLength=50,
                            maxLineGap=100)

    ###########################
    ### call the drawing fn ###

    image_with_lines = draw_lines(image, lines)

    return image_with_lines



####################
### import video ###

imported_video = cv2.VideoCapture('challenge_video.mp4')

Output_video = 'output_challenge_video.mp4'
Input_video = 'challenge_video.mp4'

imported_video = VideoFileClip(Input_video)
video_clip = imported_video.fl_image(process) # This function expects color images
video_clip.write_videofile(Output_video, audio=False)

###################################################
### apply the whole process on the video frames ###

while(imported_video.isOpened()):
    ret, frame = imported_video.read()
    frame = process(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

imported_video.release()
cv2.destroyAllWindows()

#############################################################################################################
### to get more accurate lane lines change the canny range or parameters values of hough transform method ###