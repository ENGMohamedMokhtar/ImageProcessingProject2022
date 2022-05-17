##################################
### import necessary libraries ###

from asyncore import read
import imp
from math import dist
from msilib.schema import Binary
from turtle import window_height, window_width
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from tracker import tracker
from itertools import izip_longest          # i changed the zip_longest to izip_lom=ngest and no change happened

import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np

import cv2
import pickle
import glob

###############################################################################################
### calibrate the camera depending on the chessboard images & create the undistortion model ###

# number of inner corners in x-axis & y-axis calcualated by eyes counting
num_corners_x = 9
num_corners_y = 6

# prepare object points
obj_pts = np.zeros((6*9, 3), np.float32)
obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# create arrays to store object points and image points from all images
obj_pts_arr = []    # 3d pts in real world space
img_pts_arr = []    # 2d pts in image plane

# make a list of calibration images
images = glob.glob('./camera_calibration/calibraion*.jpg')

plt.figure(figsize=(18, 12))
grid = gridspec.GridSpec(5, 4)

# set the space between axes
grid.update(wspace= 0.05, hspace= 0.15)

for idx, fname in enumerate(images):

    read_img = cv2.imread(fname)

    # convert read image to grayscale
    grayscale_img = cv2.cvtColor(read_img, cv2.COLOR_RGB2GRAY)
    
    # find the chessboard corners by cv2 fun
    ret, corners_pts = cv2.findChessboardCorners(grayscale_img, (num_corners_x, num_corners_y), None)

    # if found -> add to object pts array & images pts array
    if ret == True:
        obj_pts_arr.append(obj_pts)
        img_pts_arr.append(corners_pts)

        # draw & display corners pts by cv2 fun
        image_with_corners = cv2.drawChessboardCorners(read_img, (num_corners_x, num_corners_y), corners_pts, ret)

        write_name = 'corners found'+str(idx)+'.jpg'
        cv2.imwrite(write_name, img= read_img)

        img_plt = plt.subplot(grid[idx])
        
        plt.axis('on')
        img_plt.xticklabels([])
        img_plt.yticklabels([])

        plt.imshow(image_with_corners)
        plt.title(write_name)

        plt.axis('off')

plt.show()

# load an image for a reference
ref_image = cv2.imread('./camera_calibration/calibration1.jpg')
image_size = (ref_image.shape[1], ref_image.shape[0])

# perform the camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts_arr, img_pts_arr, image_size, None, None)

# save the camera calibration results to use later
dist_pickle = {}
dist_pickle ["mtx = "] = mtx
dist_pickle ["dist = "] = dist
pickle.dump(dist_pickle, open("pickle_data.p", "wb"))

# no need to visualize the undistortion A/B on the chessboard

# apply the undistortion on a test image
# 1- load the image
test_image = cv2.imread('./test_images/test5.jpg')
# 2- convert to RGB
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
# 3- apply undistortion
undistort_test_image = cv2.undistort(test_image, mtx, dist, None, mtx)

# no need to visualize the undistortion A/B on the test image

##############################################################################################
### define a funcition to take (image, gradient orientation, threshold min, threshold max) ###

def absolute_sobel_threshold (image, orient= 'x', thresh_min= 25, thresh_max =255):
    
    # convert to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # apply X/Y gradient with sobel() fn and take the absolute value
    if orient == 'x':
        absolute_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    
    if orient == 'y':
        absolute_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))

    # rescale to 8 bit int
    scaled_sobel = np.uint8(255*absolute_sobel/np.max(absolute_sobel))

    # create a copy from the scaled sobel
    copy_scaled_sobel = np.zeros_like(scaled_sobel)
    copy_scaled_sobel [(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1

    return copy_scaled_sobel

#######################################################
### define a funcition to color the threshold imane ###

def color_threshold (image, sthresh=(0,255), vthresh=(0,255)):
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1

    # Return the combined s_channel & v_channel binary image
    return output

######################################################
### Window_mask is a function to draw window areas ###

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

### read and make a list of test images

images = glob.glob('./test_images/*.jpg')

height = images.shape[0]
width = images.shape[1]

gidx = 0

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

#############################################################################################
### calculate the curvature and the position of the car related to the center of the lane ###

for idx, fname in enumerate(images):
    # read in read_images
    read_images = cv2.imread(fname)

    # undistort read_images
    read_images = cv2.undistort(read_images, dist, None, mtx)

    # apply threshold funcitions
    process_images = np.zeros_like(read_images[:,:,0])
    gradient_x = absolute_sobel_threshold(read_images, orient='x', thresh_min=12, thresh_max=255)
    gradient_y = absolute_sobel_threshold(read_images, orient='y', thresh_min=25, thresh_max=255)
    c_binary = color_threshold(read_images, sthresh=(100, 255), vthresh=(50, 255))
    process_images[((gradient_x == 1) & (gradient_y == 1) | (c_binary == 1))] = 255

    images_size = (read_images.shape[1], read_images.shape[0])

    src = region_of_interest(read_images, np.array([region_of_interest_vertices], np.int32,))
    offest = image_size[0]*0.25
    dst = np.float32([[offest,0], [image_size[0]-offest,0], [image_size[0]-offest, image_size[1]], [offest, image_size[1]]])

    # perform the warp prespective transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(process_images, M, images_size, flags= cv2.INTER_LINEAR)

    window_width = 25
    window_height = 80

    # set up the overall class to do the lane line tracking
    curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor=15)
    
    window_centroids = curve_centers.find_window_centroids(warped)
    
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)
        
    # points used to find the right & left lanes
    rightx = []
    leftx = []

    # Go through each level and draw the windows 
    for level in range(0,len(window_centroids)):
        # Add center value found in frame to the list of lane points per left, right
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)

        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results
    
    #fit the lane boundaries to the left, right center positions found
    yvals = range(0,warped.shape[0])
    
    res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)
    
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)
    
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx,np.int32)
    
    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    middle_marker = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    road = np.zeros_like(read_images)
    road_bkg = np.zeros_like(read_images)
    cv2.fillPoly(road,[left_lane],color=[255,0,0])
    cv2.fillPoly(road,[right_lane],color=[0,0,255])
    cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
    cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

    road_warped = cv2.warpPerspective(road,Minv,image_size,flags=cv2.INTER_LINEAR)
    road_warped_bkg= cv2.warpPerspective(road_bkg,Minv,images_size,flags=cv2.INTER_LINEAR)
    
    base = cv2.addWeighted(read_images,1.0,road_warped, -1.0, 0.0)
    result = cv2.addWeighted(base,1.0,road_warped, 1.0, 0.0)
    ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dimension
    xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dimension

    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])
    
    # Calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    # draw the text showing curvature, offset & speed
    cv2.putText(result, 'Radius of Curvature='+str(round(curverad,3))+'m ',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    plt.imshow(result, cmap='gray')
    plt.title('Final image results')
    plt.show()
    
    write_name='./test_images/tracked'+str(idx)+'.jpg'
    cv2.imwrite(write_name, result)

#################################################
### define process fun will work on the video ###

def process_image(img):
        # undistort read_images
        read_images = cv2.undistort(img, dist, None, mtx)

        warptrap = np.copy(read_images)
        cv2.line(warptrap, (int(src[0][0]), int(src[0][1])), (int(src[1][0]), int(src[1][1])), [255,0,0], 10, cv2.LINE_AA)
        cv2.line(warptrap, (int(src[1][0]), int(src[1][1])), (int(src[2][0]), int(src[2][1])), [255,0,0], 10, cv2.LINE_AA)
        cv2.line(warptrap, (int(src[2][0]), int(src[2][1])), (int(src[3][0]), int(src[3][1])), [255,0,0], 10, cv2.LINE_AA)
        cv2.line(warptrap, (int(src[3][0]), int(src[3][1])), (int(src[0][0]), int(src[0][1])), [255,0,0], 10, cv2.LINE_AA)

        # apply threshold funcitions
        process_images = np.zeros_like(read_images[:,:,0])
        gradient_x = absolute_sobel_threshold(read_images, orient='x', thresh_min=12, thresh_max=255)
        gradient_y = absolute_sobel_threshold(read_images, orient='y', thresh_min=25, thresh_max=255)
        c_binary = color_threshold(read_images, sthresh=(100, 255), vthresh=(50, 255))
        process_images[((gradient_x == 1) & (gradient_y == 1) | (c_binary == 1))] = 255

        images_size = (read_images.shape[1], read_images.shape[0])

        src = region_of_interest(read_images, np.array([region_of_interest_vertices], np.int32,))
        offest = image_size[0]*0.25
        dst = np.float32([[offest,0], [image_size[0]-offest,0], [image_size[0]-offest, image_size[1]], [offest, image_size[1]]])

        # perform the warp prespective transform
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(process_images, M, images_size, flags= cv2.INTER_LINEAR)

        window_width = 25
        window_height = 80

        # set up the overall class to do the lane line tracking
        curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor=15)
        
        window_centroids = curve_centers.find_window_centroids(warped)
        
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
            
        # points used to find the right & left lanes
        rightx = []
        leftx = []

        # Go through each level and draw the windows 
        for level in range(0,len(window_centroids)):
            # Add center value found in frame to the list of lane points per left, right
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)

            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results
        
        windowfit = np.copy(result)
        cv2.putText(windowfit, 'Sliding window results',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        warpage1 = np.copy(warpage)
        cv2.putText(warpage1, 'Bird\'s-eye View',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.line(warpage1, (int(dst[0][0]), int(dst[0][1])), (int(dst[1][0]), int(dst[1][1])), [0,0,255], 10, cv2.LINE_AA)
        cv2.line(warpage1, (int(dst[1][0]), int(dst[1][1])), (int(dst[2][0]), int(dst[2][1])), [0,0,255], 10, cv2.LINE_AA)
        cv2.line(warpage1, (int(dst[2][0]), int(dst[2][1])), (int(dst[3][0]), int(dst[3][1])), [0,0,255], 10, cv2.LINE_AA)
        cv2.line(warpage1, (int(dst[3][0]), int(dst[3][1])), (int(dst[0][0]), int(dst[0][1])), [0,0,255], 10, cv2.LINE_AA)
        
        #fit the lane boundaries to the left, right center positions found
        yvals = range(0,warped.shape[0])
        
        res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)
        
        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
        left_fitx = np.array(left_fitx,np.int32)
        
        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
        right_fitx = np.array(right_fitx,np.int32)
        
        left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        middle_marker = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2, right_fitx[::-1]-window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

        road = np.zeros_like(read_images)
        road_bkg = np.zeros_like(read_images)
        cv2.fillPoly(road,[left_lane],color=[255,0,0])
        cv2.fillPoly(road,[right_lane],color=[0,0,255])
        cv2.fillPoly(road,[inner_lane],color=[0,255,0])
        cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
        cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])
        
        #Results screen portion for polynomial fit
        road1 = np.copy(road)
        cv2.putText(road1, 'Polynomial fit',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        road_warped = cv2.warpPerspective(road,Minv,images_size,flags=cv2.INTER_LINEAR)
        road_warped_bkg= cv2.warpPerspective(road_bkg,Minv,images_size,flags=cv2.INTER_LINEAR)
        
        base = cv2.addWeighted(read_images,1.0,road_warped, -1.0, 0.0)
        result = cv2.addWeighted(base,1.0,road_warped, 1.0, 0.0)
        ym_per_pix = curve_centers.ym_per_pix # meters per pixel in y dimension
        xm_per_pix = curve_centers.xm_per_pix # meters per pixel in x dimension

        curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
        curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])
        
        # Calculate the offset of the car on the road
        camera_center = (left_fitx[-1] + right_fitx[-1])/2
        center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'

        # draw the text showing curvature, offset & speed
        cv2.putText(result, 'Radius of Curvature='+str(round(curverad,3))+'m ',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        height, width = 1080, 1920
        
        FinalScreen[0:720,0:1280] = cv2.resize(result, (1280,720), interpolation=cv2.INTER_AREA)
        FinalScreen[0:360,1280:1920] = cv2.resize(warptrap, (640,360), interpolation=cv2.INTER_AREA)
        FinalScreen[720:1080,1280:1920] = cv2.resize(warpage1, (640,360), interpolation=cv2.INTER_AREA)
        FinalScreen[720:1080,0:640] = cv2.resize(windowfit, (640,360), interpolation=cv2.INTER_AREA)
        FinalScreen[720:1080,640:1280] = cv2.resize(road1, (640,360), interpolation=cv2.INTER_AREA)

        return FinalScreen

Output_video = 'output1_tracked.mp4'
Input_video = 'project_video.mp4'
#Output_video = 'output_challenge_video.mp4'
#Input_video = 'harder_challenge_video.mp4'
#Output_video = 'output_challenge_video.mp4'
#Input_video = 'challenge_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image) # This function expects color images
video_clip.write_videofile(Output_video, audio=False)