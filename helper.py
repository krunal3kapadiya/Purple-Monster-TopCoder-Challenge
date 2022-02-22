#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:40:16 2019

PYLINT SCORE: 9.64

@author: volansys
"""
import os
import shutil
import cv2 as cv
import numpy as np
import pandas as pd

def initialize_directories(folder_name):
    '''
    creating directories
    '''
    if folder_name not in os.listdir(os.getcwd()):
        os.mkdir(os.getcwd()+'/'+folder_name)
    else:
        shutil.rmtree(os.getcwd()+'/'+folder_name)
        os.mkdir(os.getcwd()+'/'+folder_name)

def clean_directories():
    '''
    cleaning and recreating the directories
    '''
    initialize_directories('saved_csv')
    initialize_directories('saved_csv/indexed_data')
    initialize_directories('before')
    initialize_directories('after')
    initialize_directories('cropped')
    initialize_directories('cropped/before')
    initialize_directories('cropped/after')

def get_scaled_image(image, size):
    '''
    will return scaled image
    here I taken 500 constant,
    because averege size of the cropped square pixel is 500
    '''
    max_dimension = max(image.shape)
    scale = size/max_dimension
    scaled_image = cv.resize(image, None, fx=scale, fy=scale) # pylint: disable=no-member
    return scaled_image

def find_seeds(image):
    '''
    detecting contours from the image
    '''
    image_blur = cv.GaussianBlur(image, (7, 7), 0) # pylint: disable=no-member
    image_blur_hsv = cv.cvtColor(image_blur, cv.COLOR_RGB2HSV) # pylint: disable=no-member

    # min yellow colour array
    min_yellow = np.array([0, 60, 55])
    max_yellow = np.array([120, 140, 155])
    # layer
    mask1 = cv.inRange(image_blur_hsv, min_yellow, max_yellow) # pylint: disable=no-member
    mask1 = cv.convertScaleAbs(mask1) # pylint: disable=no-member
    # creating another mask to detect yellow seeds
    # 170-180 hue
    min_yellow = np.array([160, 0, 0])
    max_yellow = np.array([180, 0, 0])
    mask2 = cv.inRange(image_blur_hsv, min_yellow, max_yellow) # pylint: disable=no-member
    # Combine masks
    mask = mask1 + mask2
    # Clean up
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15)) # pylint: disable=no-member
    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel) # pylint: disable=no-member
    # erosion followed by dilation. It is useful in removing noise
    mask_clean = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernel) # pylint: disable=no-member
    contours, _ = cv.findContours(mask_clean, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # pylint: disable=no-member
    return contours

def save_seeds_from_images(input_image, 
                 output_image_path, 
                 image_index_number, 
                 save_seeds_images):
    '''
    saving seeds and images from contours
    '''
    list_position_x = []
    list_position_y = []
    list_seed_width = []
    list_seed_height = []
    list_angle = []
    list_image_name = []
    list_contours = []
    
    input_image = get_scaled_image(input_image, 500)
    contours = find_seeds(input_image)
    for i, contour in enumerate(contours):
        x_point, y_point, width, height = cv.boundingRect(contour) # pylint: disable=no-member
        roi = input_image[y_point-5:y_point+height+5, x_point-5:x_point+width+5]
        # if shape is more than 30 than it is not valid seed
        if (max(roi.shape) < 30):
            if save_seeds_images:
                image_name = '{}_img_{}.jpg'.format(image_index_number, i)                
                cv.imwrite(output_image_path+'/'+image_name, roi) # pylint: disable=no-member
                list_image_name.append(image_name)
            # for contour to fit in ellipse there is minimum 5 size required
            if len(contour) >= 5:
                ellipse = cv.fitEllipse(contour)# pylint: disable=no-member
                ((x_position, y_position), (seed_width, seed_height), angle) = ellipse
                list_position_x.append(x_position)
                list_position_y.append(y_position)
                list_seed_width.append(seed_width)
                list_seed_height.append(seed_height)
                list_angle.append(angle)
                list_contours.append(contour)
    df_seed_data = pd.DataFrame({'image_name': pd.Series(list_image_name),
                                 'position_x':pd.Series(list_position_x),
                                 'position_y':pd.Series(list_position_y),
                                 'seed_width': pd.Series(list_seed_width),
                                 'seed_height': pd.Series(list_seed_height),
                                 'angle':pd.Series(list_angle)})
    return df_seed_data, input_image, list_contours

def crop_images(original_image):
    '''
    cropping images
    '''
    input_image = get_scaled_image(original_image, 800)
    output_image = input_image.copy()#copying to create mask
    gray = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY) # pylint: disable=no-member
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 500) # pylint: disable=no-member
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x_point, y_point, radius) in circles:
            x_point = (x_point-radius).clip(min=0)
            y_point = (y_point-radius).clip(min=0)
            height = 2*radius
            output_image = output_image[y_point:y_point+height, x_point:x_point+height]
    return output_image

def set_wd(directory_name, original_directory):
    '''
    set working directory
    '''
    if directory_name is not None and os.path.isdir(directory_name):
        return os.chdir(directory_name)
    else:
        return os.chdir(original_directory)
