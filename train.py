#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:40:16 2019
PYLINT SCORE: 8.79
Disabled cv, because there is opencv not supporting lint
@author: volansys
"""

import os
from functools import reduce
import glob
import pickle as pkl
import argparse
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import helper

AP = argparse.ArgumentParser()
AP.add_argument("--datasetPath", required=False, help="Enter dataset path")
AP.add_argument("--outputPath", required=False, help="Enter output path")

ARGS = vars(AP.parse_args())

dataset_path = ARGS['datasetPath']
output_path = ARGS['outputPath']

original_dir = os.getcwd()

def images_list_to_csv(image_list):
    '''
    seperating two columns and saving list of images in CSV file
    '''
    first_day_images = []
    second_day_images = []
    for image in image_list:
        if image.__contains__('24Hours'):
            second_day_images.append(image)
        else:
            first_day_images.append(image)
    df_image_list = pd.DataFrame({"first_day":pd.Series(sorted(first_day_images)),
                                  "second_day":pd.Series(sorted(second_day_images))})
    df_image_list.to_csv('saved_csv/indexed_images.csv', index=False)
    return df_image_list

def perform_operations(original_file_name,
                       output_file_path,
                       index_number):
    '''
    common method to perform crop and detecting seeds
    '''
    # original image
    helper.set_wd(dataset_path, original_dir)
    image = cv.imread('images/'+original_file_name) # pylint: disable=no-member
    # cropped image as per plate
    helper.set_wd(output_path, original_dir)
    plate_image = helper.crop_images(image)
    # save image
    # e.g., cropped/before/original_filename.jpeg
    image_name = 'cropped/'+output_file_path+'/{}'.format(original_file_name)
    cv.imwrite(image_name, plate_image) # pylint: disable=no-member

    # detecting seeds in the square image
    input_image = cv.imread(image_name) # pylint: disable=no-member
    df_seed_data, _, _ = helper.save_seeds_from_images(input_image, output_file_path, index_number, True)
    output_seed_csv_name = 'saved_csv/indexed_data/seed_{}_data_{}.csv'.format(output_file_path, index_number)
    df_seed_data.to_csv(output_seed_csv_name, index=False)

def is_seed_is_purple(image):
    '''
    if black (purple) is detected in seeds then it will give false, else true
    '''
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV) # pylint: disable=no-member
    lower_purple = np.array([0, 0, 0])
    upper_purple = np.array([90, 90, 90])
    mask = cv.inRange(hsv, lower_purple, upper_purple) # pylint: disable=no-member
    return 255 in mask

def main():
    '''
    this is main function, it will clear the directories
    - finding square of the circle and crop it
    - will find the seed from the cropped circle images

    CHANGABLE VARIABLE WILL BE END RANGE -
    CURRENTLY IT IS SET AS 5 AFTER END TO END
    OPERATIONS COMPLETES SET THE RANGE AS
    df_image_list.shape[0]
    '''
    print('creating directories...')
    helper.set_wd(output_path, original_dir)
    helper.clean_directories()
    print('reading images...')
    helper.set_wd(dataset_path, original_dir)
    images_list = pd.DataFrame(os.listdir('images/'))[0]
    helper.set_wd(output_path, original_dir)
    df_image_list = images_list_to_csv(images_list)
    helper.set_wd(output_path, original_dir)
    for i in range(0, df_image_list.shape[0]): #initially runing operations only for 10 images
        
        original_image_name = df_image_list.iloc[i, 0]
        print('performing operations for {}...'.format(original_image_name))
        after_image_name = df_image_list.iloc[i, 1]
        # performing operations in each images
        perform_operations(original_image_name, 'before', i)
        perform_operations(after_image_name, 'after', i)

    # saving images that are detected common in both folder and
    # saving it in csv file
    common_images_list = list(set(os.listdir('before/')) and set(os.listdir('after/')))
    identical_images_df = pd.DataFrame({"identical_images": sorted(common_images_list)})

    # now selecting the seeds that converted into purple in next day and saving it
    is_purple = []
    empty_seeds = []
    image_name_list = []
    for i in range(identical_images_df.shape[0]):
        image_name = identical_images_df.iloc[i, 0]
        print('detecting changed seeds for {}...'.format(image_name))
        image = cv.imread('after/'+ identical_images_df.iloc[i, 0]) # pylint: disable=no-member
        if image is not None:
            is_purple.append(is_seed_is_purple(image))
            image_name_list.append(image_name)
        else:
            empty_seeds.append(image_name)
    print('saving data...')
    purpuled_images = pd.DataFrame({'monster': pd.Series(is_purple),
                                    'image_name': pd.Series(image_name_list)})
    purpuled_images.to_csv('saved_csv/seeds_converted.csv', index=False)

    ######### merging all the columns in data
    all_files = glob.glob('saved_csv/indexed_data/*.csv')
    files_list = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        files_list.append(df)
    df_final = pd.concat(files_list, axis=0, ignore_index=True)
    df_final.to_csv('saved_csv/final_data.csv', index=False)
    ######### merged
    #### creating the main csv file
    main_df = reduce(lambda x, y: pd.merge(x, y, on=['image_name'], how='outer'),
                     [df_final, purpuled_images])
    main_df.dropna(inplace=True)
    main_df.to_csv('saved_csv/maindf.csv', index=False)
    main_df = pd.read_csv('saved_csv/maindf.csv')
    X = main_df.iloc[:, 1:-2]
    y = main_df.iloc[:, 6]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    pkl.dump(classifier, open('classifier.pkl', 'wb'))
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('accuracy score of model is {}'.format(score))
    ##############################
main()
