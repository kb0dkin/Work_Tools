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

    # parse the annotations, split into different views, put output json in image_directory
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
                    crop_fn = os.path.join(subdir,f'{os.path.splitext(image_name)[0]}_{key}.png')
                    cv2.imwrite(crop_fn, im_crop)





# flip views to account for the whole "mirror" thing
def view_flipper(image, view_name:str):
    if view_name.lower() == 'south': # if south flip LR
        return image[:,::-1,:]
    elif view_name.lower() == 'north': # if north flip top to bottom
        return image[::-1,:,:]
    elif view_name.lower() == 'east': # if east transpose
        return image.transpose((1,0,2))
    elif view_name.lower() == 'west': # the "anti-transpose" per Math Overflow haha
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
            
            if 'annotatedResult' in data.keys():
            # if 'test-3d-data-20240213' in data.keys():
                data = data['annotatedResult']
                # data = data['test-3d-data-20240213']

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
                    bounded_df = data_df.iloc[(data_df.x.between(bound[0], bound[2]) & data_df.y.between(bound[1], bound[3])).values]


                    # split label into a dict
                    label_dict,(width, height) = make_dict(bounded_df, bound, bound_name)
                    
                    # assemble the full entry
                    entry_dict = {
                                'image': os.path.join(image_dir, subimage),
                                'height': height,
                                'width': width,
                                'ann': label_dict,
                                'ann_label': bounded_df['label'].unique(),
                                'frame_id': os.path.join(image_dir, subimage),
                    }

                    label_list.append(entry_dict)

    with open(os.path.join(image_dir, 'processed_keypoints.json'), 'w') as fid:
        json.dump(label_list, fid)
    # return label_list


def make_dict(bounded_df, bound, bound_name):


    # pull x and y into a list of lists -- grouped by worker
    label_df,(width, height) = label_view_parser(bounded_df, bound, bound_name)
    labels_x = label_df.groupby('worker_id')['x'].apply(lambda x: x.values.tolist()).values
    labels_y = label_df.groupby('worker_id')['y'].apply(lambda x: x.values.tolist()).values

    # meds
    med_x = label_df.groupby('label')['x'].median()
    med_y = label_df.groupby('label')['y'].median()

    # means
    mu_x = label_df.groupby('label')['x'].mean()
    mu_y = label_df.groupby('label')['y'].mean()

    # stdiance
    std_x = label_df.groupby('label')['x'].std()
    std_y = label_df.groupby('label')['y'].std()

    # bounding box 
    # length for each side will be 0.6 * dist between extrema of medians
    Bx_off = abs(max(med_x)-min(med_x)) * 0.3 # offsets for bounding box
    By_off = abs(max(med_y)-min(med_y)) * 0.3 # offsets for bounding box
    B_xmin = max([0, min(med_x) - Bx_off]) # don't go below 0
    B_xmax = min([bound[2], max(med_x) + Bx_off]) # don't go above width of image
    B_ymin = max([0, min(med_y) - By_off])
    B_ymax = min([bound[3], max(med_y) + By_off])
    B_area = (B_xmax - B_xmin)*(B_ymax - B_ymin)


    label_dict = {'X': labels_x.tolist(),
                  'Y': labels_y.tolist(),
                  'bbox': np.array([B_xmin, B_xmax, B_ymin, B_ymax]).tolist(),
                  'med': np.array([med_y, med_x]).tolist(),
                  'mu': np.array([mu_y, mu_x]).tolist(),
                  'std': np.array([std_y, std_x]).tolist(),
                  'area': [B_area]
                  }
    
    print(label_dict)
    

    return label_dict, (width, height)
    

# adjust the label points to account for rotating/reflecting the mirrors
def label_view_parser(bounded_df:pd.DataFrame, bound, bound_name:str):
    label_df = bounded_df.copy() # label_df == corrected dataframe

    if bound_name.lower() == 'center': # fit to boundaries
        label_df.x -= bound[0]
        label_df.y -= bound[1]
        # width and height -- to swap if we rotate the image
        width = bound[2] - bound[0]
        height = bound[3] - bound[1]
    
    if bound_name.lower() == 'north': # fit and flip vertically
        label_df.x -= bound[0]
        label_df.y = bound[3] - label_df.y
        # width and height -- to swap if we rotate the image
        width = bound[2] - bound[0]
        height = bound[3] - bound[1]

    if bound_name.lower() == 'south': # fit and flip horizontally
        label_df.x = bound[2] - label_df.x
        label_df.y -= bound[1]
        # width and height -- to swap if we rotate the image
        width = bound[2] - bound[0]
        height = bound[3] - bound[1]
    
    if bound_name.lower() == 'east': # transpose cropped x and y
        temp_y = label_df.x - bound[0] # future y == cropped (old) x
        label_df.x = label_df.y - bound[1] # x == cropped (old) y
        label_df.y = temp_y # should be a pd series
        # width and height -- to swap if we rotate the image
        height = bound[2] - bound[0]
        width = bound[3] - bound[1]

    if bound_name.lower() == 'west': # "anti-transpose"
        # I think this is x = maxY - y, y = maxX - x
        temp_y = bound[2]-label_df.x
        label_df.x = bound[3] - label_df.y
        label_df.y = temp_y
        # width and height -- to swap if we rotate the image
        height = bound[2] - bound[0]
        width = bound[3] - bound[1]


    return label_df, (width, height)


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
