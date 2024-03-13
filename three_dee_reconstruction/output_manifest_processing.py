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


    # split images and labels into different views, save images and json
    splitter(manifest_path, boundaries, image_directory)



# boundaries file list processor
def bound_list2dict(bound_list:List[int]):
    # view_names = ['North','South','East','West','Center']
    view_names = ['West', 'East', 'North', 'Center', 'South'] # same order as multiview_aws_preparation
    bound_dict = dict(zip(view_names,bound_list))

    return bound_dict




# split the images and the annotations
def splitter(manifest_path:str, boundaries:dict, image_dir:str):
    # # create a list -- double list comprehension to flatten it
    # image_list = [item for file in ['*.jpg','*.tiff','*.png'] for item in glob.glob(os.path.join(image_directory,file))]
    
    # make a directory for the split files
    subdir_below = os.path.join(image_dir,'below')
    subdir_side = os.path.join(image_dir, 'side')
    if not os.path.exists(subdir_below):
        os.mkdir(subdir_below)  # create a subdirectory
    if not os.path.exists(subdir_side):
        os.mkdir(subdir_side)  # create a subdirectory

    # open the manifest
    with open(manifest_path, 'r') as manifest_fid:
        # list of dicts for the labels
        label_list_below = []
        label_list_side = []

        for line in manifest_fid.readlines():
            data = json.loads(line)
            
            # pull out image name, get the boundaries for this image
            image_name = os.path.split(data['source-ref'])[-1]
            boundary = boundaries[image_name]

            # do we have the image in the image_dir?
            if not os.path.exists(os.path.join(image_dir, image_name)):
                print(f'{image_name} not found in {image_dir}')
                continue
            
            # start opening the images
            image = cv2.imread(os.path.join(image_dir, image_name)) # open the image

            # is the image found in the boundary list?
            if image_name not in boundaries.keys(): # skip this image if it's not in the boundary list
                print(f'{image_name} not found in boundaries.txt')
                continue
        
        
            if 'annotatedResult' in data.keys():
                data = data['annotatedResult']

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

                    if bound_name.lower() == 'center':
                        subdir = subdir_below
                    else:
                        subdir = subdir_side

                    # pop out the appropriate labels
                    bounded_df = data_df.iloc[(data_df.x.between(bound[1], bound[3]) & data_df.y.between(bound[0], bound[2])).values]

                    # skip the rest of this if there isn't any labeling 
                    # in the view
                    if len(bounded_df) == 0:
                        continue

                    # split label into a dict
                    label_dict,(width, height) = make_dict(bounded_df, bound, bound_name)
                    
                    # assemble the full entry
                    entry_dict = {
                                'image': os.path.join(subdir, subimage),
                                'height': height,
                                'width': width,
                                'ann_label': bounded_df['label'].unique().tolist(),
                                'frame_id': os.path.join(subdir, subimage),
                                'ann_black': label_dict,
                    }


                    # split out the boundary portion of this image
                    im_crop = image[bound[0]:bound[2],bound[1]:bound[3]] # crop the image
                    im_crop = view_flipper(im_crop, bound_name).astype(np.uint8)

                    # save 'center' views separately from 'side' views
                    if bound_name.lower() == 'center':
                        cv2.imwrite(os.path.join(subdir_below, subimage), im_crop)
                        label_list_below.append(entry_dict)
                    else:
                        cv2.imwrite(os.path.join(subdir_side, subimage), im_crop)
                        label_list_side.append(entry_dict)

            # if there aren't any labels in the output manifest...
            else:
                print(f'No data for {image_name} found in output manifest')


    # save the label lists
    with open(os.path.join(subdir_below, 'processed_keypoints.json'), 'w') as fid:
        json.dump(label_list_below, fid)
    with open(os.path.join(subdir_side, 'processed_keypoints.json'), 'w') as fid:
        json.dump(label_list_side, fid)



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



# create a label dict
def make_dict(bounded_df:pd.DataFrame, bound, bound_name:str):

    # pull x and y into a list of lists -- grouped by worker
    label_df,(width, height) = label_view_parser(bounded_df, bound, bound_name)
    labels_x = label_df.groupby('worker_id')['x'].apply(lambda x: np.divide(x.values,width).tolist()).values
    labels_y = label_df.groupby('worker_id')['y'].apply(lambda x: np.divide(x.values,height).tolist()).values

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
    Bx_off = abs(np.nanmax(med_x)-np.nanmin(med_x)) * 0.3 # offsets for bounding box
    By_off = abs(np.nanmax(med_y)-np.nanmin(med_y)) * 0.3 # offsets for bounding box
    B_xmin = max([0, min(med_x) - Bx_off]) # don't go below 0
    B_xmax = min([width, max(med_x) + Bx_off]) # don't go above width of image
    B_ymin = max([0, min(med_y) - By_off])
    B_ymax = min([height, max(med_y) + By_off])
    B_area = (B_xmax - B_xmin)*(B_ymax - B_ymin)

    # print(labels_x)
    # exit()

    label_dict = {'X': labels_x.tolist(),
                  'Y': labels_y.tolist(),
                  'bbox': np.array([B_xmin/width, B_xmax/width, B_ymin/height, B_ymax/height]).tolist(),
                  'med': np.array([med_y/height, med_x/width]).tolist(),
                  'mu': np.array([mu_y/height, mu_x/width]).tolist(),
                  'std': np.array([std_y/height, std_x/width]).tolist(),
                  'area': B_area
                  }
    
    return label_dict, (width, height)
    

# adjust the label points to account for rotating/reflecting the mirrors
def label_view_parser(bounded_df:pd.DataFrame, bound, bound_name:str):
    label_df = bounded_df.copy() # label_df == corrected dataframe

    if bound_name.lower() == 'center': # fit to boundaries
        label_df.x -= bound[1]
        label_df.y -= bound[0]
        # width and height -- to swap if we rotate the image
        height = bound[2] - bound[0]
        width = bound[3] - bound[1]
    
    if bound_name.lower() == 'north': # fit and flip vertically
        label_df.x -= bound[1]
        label_df.y = bound[2] - label_df.y
        # width and height -- to swap if we rotate the image
        height = bound[2] - bound[0]
        width = bound[3] - bound[1]

    if bound_name.lower() == 'south': # fit and flip horizontally
        label_df.x = bound[3] - label_df.x
        label_df.y -= bound[0]
        # width and height -- to swap if we rotate the image
        height = bound[2] - bound[0]
        width = bound[3] - bound[1]
    
    if bound_name.lower() == 'east': # transpose cropped x and y
        temp_y = label_df.x - bound[1] # future y == cropped (old) x
        label_df.x = label_df.y - bound[0] # x == cropped (old) y
        label_df.y = temp_y # should be a pd series
        # width and height -- to swap if we rotate the image
        height = bound[2] - bound[0]
        width = bound[3] - bound[1]

    if bound_name.lower() == 'west': # "anti-transpose"
        # I think this is x = maxY - y, y = maxX - x
        temp_y = bound[3]-label_df.x
        label_df.x = bound[2] - label_df.y
        label_df.y = temp_y
        # width and height -- to swap if we rotate the image
        height = bound[2] - bound[0]
        width = bound[3] - bound[1]


    return label_df, (width, height)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_path')
    parser.add_argument('image_directory', default='.')

    args = parser.parse_args()

    output_manifest_processing(args.manifest_path, args.image_directory)
