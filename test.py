#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:47:42 2019
PYLINT SCORE: 9.21
@author: krunal
"""

import pickle
import os
import argparse
import pandas as pd
import cv2 as cv
import helper

AP = argparse.ArgumentParser()
AP.add_argument("--datasetPath", required=False, help="Enter dataset path")
AP.add_argument("--outputPath", required=False, help="Enter output path")

ARGS = vars(AP.parse_args())

dataset_path = ARGS['datasetPath']
output_dir = ARGS['outputPath']

original_dir = os.getcwd()

def filter_input_data(image_name):
    '''
    filtering input data
    '''
    helper.set_wd(dataset_path, original_dir)
    image = cv.imread('images/'+image_name) # pylint: disable=no-member
    helper.set_wd(output_dir, original_dir)
    # step 1 crop image
    output_image = helper.crop_images(image)
    return helper.save_seeds_from_images(output_image, None, None, False)

def draw_and_save_images(input_image, image_name, contours, result):
    '''
    drawing the circle on the predicted images
    '''
    image_with_ellipse = input_image.copy()
    for i, contour in enumerate(contours):
        if result[i]:
            # checking contour array size to fit in ellipse
            # checking that if contour is not grater than seed size
            if len(contour) >= 5 and max(contour.shape) < 50:
                ellipse = cv.fitEllipse(contour) # pylint: disable=no-member
                cv.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2) # pylint: disable=no-member
    cv.imwrite('result/'+image_name.split(sep='.')[0]+'_pred.jpeg', image_with_ellipse) # pylint: disable=no-member

def main():
    '''
    performing, predicting values from saved modela and saving values
    '''
    helper.set_wd(output_dir, original_dir)
    helper.initialize_directories('result')
    df_main = pd.read_csv('saved_csv/indexed_images.csv')
    classifier = pickle.load(open('classifier.pkl', 'rb'))
    for image_name in df_main.iloc[:, 0]:
        print("processing for {} ...".format(image_name))
        df_seed_data, input_image, contours = filter_input_data(image_name)
        input_data = df_seed_data.iloc[:, 1:5]
        if input_data.shape[0] != 0:
            result = classifier.predict(input_data)
            draw_and_save_images(input_image, image_name, contours, result)

main()
