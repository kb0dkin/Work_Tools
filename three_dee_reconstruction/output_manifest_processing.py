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
        for line in manifest_fid.readlines():
            data = json.loads(line)
            if 'annotatedResult' in data.keys():
                data = data['annotatedResult'] # 
                # image name
                image_name = os.path.split(json.loads(line)['source-ref'])[-1]
                boundary = boundaries[image_name]

                # we'll create a pandas array for each view
                labels = pd.DataFrame()

                # work through each of the workers
                for worker_entry in data['annotationsFromAllWorkers']:

                    worker_data = pd.DataFrame.from_dict(eval(worker_entry['annotationData']['content'])['annotatedResult']['keypoints'])
                    label_view_parser(worker_data, boundary)
                    # worker_data.loc[:,'view'] = label_view_parser(worker_data, boundary)
                    # print(worker_data)


# pull 
def label_view_parser(label_pd, boundaries:dict):
    label_view = []

    # find which view the label is in:
    for view,boundary in boundaries.items():
        print(f'{label_pd.x}: {label_pd.x < boundary[0]}')
        # if label_pd.x < boundary[0] and label_pd.x > boundary[2] and label_pd.y > boundary[1] and label_pd.y > boundary[3]:
        #     label_view.append(view)

    return label_view



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
