from turtle import width
from matplotlib import image        # import image plot type from the lib
import matplotlib.pylab as plt      # library for plot (as plt -> used for create a shortcut instead of the long name)
import cv2                          # library for computer vision
import numpy as np                  # library for mathematical operations & all powerful equations

##########################################
### import image -> convert its format ###

image = cv2.imread('straight_lines1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

##############################
### image height and width ###

print(image.shape)              # shape matrix is [height, width, no. of channels]
height = image.shape[0]         # 720      -> shape[0] = 720
width = image.shape[1]          # 1200     -> shape[1] = 1200

###########################################
### define the region of interest edges ###

region_of_interest_vertices = [
    # (width, height)
    (200, height-60),
    (610, 430),
    (675, 430),
    (1100, height-60)
]

##########################################################################
### define a funcition to mask the region of interest from input image ###

def region_of_interest (input_image, vertices):
    mask = np.zeros_like(input_image)               
    
    # channel_count = input_image.shape[2]            # no. of colors channels in the image
    match_mask_color = 255                            # make it one color has scale 255

    cv2.fillPoly(mask, vertices, match_mask_color)

    after_masking_image = cv2.bitwise_and(input_image, mask)

    return after_masking_image

####################################################################
### convert the input image to grayscale then apply canny filter ###

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 200, 255)       # what's canny ?! canny is a filter of converting the image to lines and strokes

###########################################################################
### call the fn and give it the canny image & interest edges identified ###

masked_image_for_work = region_of_interest(
    canny_image,
    np.array([region_of_interest_vertices], np.int32,)
)

################################################################################
### get lines from the masked_image_for_work by using hough transform method ###
### parameters:
##### 1- image source
##### 2- rho -> the distance resolution of the accumlater in pixels
##### 3- theta -> angle resolution of accumlater in radians
##### 4- threshold -> ????????????????????????????????????????????????????????????????????????????????
##### 5- lines -> ?????????????????????????????????????????????????????????????????????????/
##### 6- minimum lenght of objects to consider it a line
##### 7- maximum gap between two lines

lines = cv2.HoughLinesP(masked_image_for_work,
                        rho=6,
                        theta=np.pi/60,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=20,
                        maxLineGap=165)

####################################################
### create a fn to draw lines on the input image ###

#take two parameters (original_image - lines created)
def draw_lines (img, lines):
    img = np.copy(img)

    #create a blank image with the same dimensions of the original image
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness= 5)
    
    #merge original image with the blank image
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

###########################
### call the drawing fn ###

image_with_lines = draw_lines(image, lines)

plt.imshow(image_with_lines)
plt.show()