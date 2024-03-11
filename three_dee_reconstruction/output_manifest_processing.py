import argparse
import os
import json
import glob
from typing import List
import cv2
import numpy as np
import pandas as pd

from pprint import pprint




# pull in a multi-view output manifest from AWS, then output the clipped views and 
# associated labels into something that MARS can take in
#
# I've only written this for the multi-view with 4 mirror setup. The code will need to be
# adjusted to work on any other setups

def output_manifest_processing(manifest_path:str, image_directory:str, output_type:str = 'MARS'):

    # input parsing -- does everything we need exist?
    # does the manifest exist?
    if not os.path.exists(manifest_path):
        print(f'Manifest file {manifest_path} does not exist')
        return -1

    # does the image directory exist?
    if not os.path.exists(image_directory):
        print(f'Directory {image_directory} does not exist')
        return -1

    # is there a boundaries.txt file in there to split the images apart?
    if not os.path.exists(os.path.join(image_directory,'boundaries.txt')):
        print(f"boundaries.txt file does not exist. Place into {image_directory}")
        return -1

    # load the boundaries file into a dictionary
    with open(os.path.join(image_directory,'boundaries.txt')) as fid:
        boundaries = {line.split(':')[0]:bound_list2dict(eval(line.split(':')[1])) for line in fid.readlines()}

    # split the images according to the boundaries
    # image_splitter(image_directory, boundaries)

    # parse the annotations, split into different views
    annot_splitter(manifest_path, boundaries, image_directory)



# boundaries file list processor
def bound_list2dict(bound_list:List[int]):
    # view_names = ['North','South','East','West','Center']
    view_names = ['West', 'East', 'North', 'Center', 'South'] # same order as multiview_aws_preparation
    bound_dict = dict(zip(view_names,bound_list))

    return bound_dict


# split the images and flip/rotate the side views as needed
def image_splitter(image_directory:str, boundaries:dict):
    # create a list -- double list comprehension to flatten it
    image_list = [item for file in ['*.jpg','*.tiff','*.png'] for item in glob.glob(os.path.join(image_directory,file))]

    # make a directory for the split files
    subdir = os.path.join(image_directory,'split')
    if not os.path.exists(subdir):
        os.mkdir(subdir)  # create a subdirectory

    # iterate through each image
    for image_fn in image_list:
        image = cv2.imread(image_fn) # open the image
        im_basename = os.path.split(image_fn)[-1] # for dictionary keys, cropped image names etc
        
        if im_basename not in boundaries.keys(): # skip this image if it's not in the boundary list
            print(f'{im_basename} not found in boundaries.txt')
            continue

        boundary = boundaries[im_basename] # get the boundaries for this image
        for key,bound in boundary.items(): # for each boundary
            im_crop = image[bound[0]:bound[2],bound[1]:bound[3]] # crop the image
            im_crop = view_flipper(im_crop, key).astype(np.uint8)
            crop_fn = os.path.join(subdir,f'{os.path.splitext(im_basename)[0]}_{key}.png')
            cv2.imwrite(crop_fn, im_crop)


# splitter - do both the images and the labels all at once
def splitter(image_directory:str, boundaries:dict, manifest_path):
    # create a list -- double list comprehension to flatten it
    image_list = [item for file in ['*.jpg','*.tiff','*.png'] for item in glob.glob(os.path.join(image_directory,file))]

    # make a directory for the split files
    subdir = os.path.join(image_directory,'split')
    if not os.path.exists(subdir):
        os.mkdir(subdir)  # create a subdirectory


    with open(manifest_path, 'r') as manifest_fid:
        for line in manifest_fid.readlines():
            data = json.loads(line)
            
            # pull out image name, get the boundaries for this image
            image_name = os.path.split(data['source-ref'])[-1]
            
            # if 'annotatedResult' in data.keys():
            if 'test-3d-data-20240213' in data.keys(): # odd key name for the test dataset. Should use the other for the main dataset
                # data = data['annotatedResult']
                data = data['test-3d-data-20240213']

                # find the image path in the list
                image_path = [item for item in image_list if image_name in item]
                if len(image_path) != 1: # if this matches zero images or more than one image...
                    print(f'Found {len(image_path)} that match {image_name}')
                    continue
                image = cv2.imread(image_path) # open the image

                # check if image is found in the boundaries.txt file
                if image_name not in boundaries.keys(): # skip this image if it's not in the boundary list
                    print(f'{image_name} not found in boundaries.txt')
                    continue
                boundary = boundaries[image_name] # get the boundaries for this image

                # work through each of the workers
                for worker_entry in data['annotationsFromAllWorkers']:

                    worker_data = pd.DataFrame.from_dict(eval(worker_entry['annotationData']['content'])['annotatedResult']['keypoints'])
                    label_view_parser(worker_data, boundary)
                    # worker_data.loc[:,'view'] = label_view_parser(worker_data, boundary)
                    # print(worker_data)

                # split the images into different  
                for key,bound in boundary.items(): # for each boundary
                    im_crop = image[bound[0]:bound[2],bound[1]:bound[3]] # crop the image
                    im_crop = view_flipper(im_crop, key).astype(np.uint8)
                    crop_fn = os.path.join(subdir,f'{os.path.splitext(im_basename)[0]}_{key}.png')
                    cv2.imwrite(crop_fn, im_crop)





# flip views to account for the whole "mirror" thing
def view_flipper(image, view_name:str):
    if view_name == 'South': # if south flip LR
        return image[:,::-1,:]
    elif view_name == 'North': # if north flip top to bottom
        return image[::-1,:,:]
    elif view_name == 'East': # if east transpose
        return image.transpose((1,0,2))
    elif view_name == 'West': # the "anti-transpose" per Math Overflow haha
        return image[::-1].transpose(1,0,2)[::-1]
    else: # should be center otherwise
        return image


# split annotations per the boundaries
def annot_splitter(manifest_path:str, boundaries:dict, image_dir:str):
    
    with open(manifest_path, 'r') as manifest_fid:
        # list of dicts for the labels
        label_list = []

        for line in manifest_fid.readlines():
            data = json.loads(line)
            
            # pull out image name, get the boundaries for this image
            image_name = os.path.split(data['source-ref'])[-1]
            boundary = boundaries[image_name]
            
            # if 'annotatedResult' in data.keys():
            if 'test-3d-data-20240213' in data.keys():
                # data = data['annotatedResult']
                data = data['test-3d-data-20240213']

                # # skip rest of loop for the moment
                # print(data)
                # continue

                # entry for each view -- will turn into a list of dicts later to put
                # into the main list but we need to be able to revisit the images
                # for multiple workers
                entries = {}

                # append data from all of the workers together
                data_df = pd.DataFrame()
                for worker_entry in data['annotationsFromAllWorkers']:
                    # parse out the info
                    worker_df = pd.DataFrame.from_dict(eval(worker_entry['annotationData']['content'])['annotatedResult']['keypoints'])
                    worker_df['worker_id'] = worker_entry['workerId']
                    data_df = pd.concat([data_df, worker_df], ignore_index = True)

                # create a sublist and run through each view.
                for bound_name, bound in boundary.items():
                    # which subimage?
                    subimage = os.path.splitext(image_name)[0] + '_' + bound_name + '.png'

                    # pop out the appropriate labels
                    # print(data_df.x.between(bound[0], bound[2]) & data_df.y.between(bound[1], bound[3]))
                    bounded_df = data_df.iloc[(data_df.x.between(bound[0], bound[2]) & data_df.y.between(bound[1], bound[3])).values]

                    # pull x and y into a list of lists -- grouped by worker
                    labels_x = bounded_df.groupby('worker_id')['x'].apply(list).values
                    labels_y = bounded_df.groupby('worker_id')['y'].apply(list).values

                    # means
                    means_x = bounded_df.groupby('worker_id')['x'].mean()
                    means_y = bounded_df.groupby('worker_id')['y'].mean()

                    # variance
                    vars_x = bounded_df.groupby('worker_id')['x'].var()
                    vars_y = bounded_df.groupby('worker_id')['y'].var()
                    
                    print(f'x: {labels_x}')
                    print(f'y: {labels_y}')

                    # worker_data = pd.DataFrame.from_dict(eval(worker_entry['annotationData']['content'])['annotatedResult']['keypoints'])
                    # label_view_parser(worker_data, boundary)
                    # worker_data.loc[:,'view'] = label_view_parser(worker_data, boundary)
                    # print(worker_data)


# split the labels by view, and manipulate as needed 
def label_view_parser(label_pd, boundaries:dict):
    print(label_pd)

    for b_name, boundary in boundaries.items():
        label_pd[b_name] = label_pd.x.between(boundary[0], boundary[2]) & label_pd.y.between(boundary[1], boundary[3])

    print(label_pd)

    # for each label set (will have labels from multiple views):
    # for label in label_pd:




    # # find which view the label is in:
    # for view,boundary in boundaries.items():
    #     within = (label_pd.x > boundary[0]) and (label_pd.x < boundary[2]) and (label_pd.y > boundary[1]) and (label_pd.y < boundary[3])
        
    #     # if label_pd.x < boundary[0] and label_pd.x > boundary[2] and label_pd.y > boundary[1] and label_pd.y > boundary[3]:
    #     #     label_view.append(view)

    # return label_view



# turn it into a mars json
def convert_to_mars():
    pass



# turn it into something for sleap
def convert_to_sleap():
    pass



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_path')
    parser.add_argument('image_directory', default='.')

    args = parser.parse_args()

    output_manifest_processing(args.manifest_path, args.image_directory)
