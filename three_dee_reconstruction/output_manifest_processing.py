import argparse
import os
import json
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

def output_manifest_processing(manifest_path:str, image_directory:str, keypoint_file:str, output_type:str = 'MARS'):

    # input parsing -- does everything we need exist?
    # does the manifest exist?
    if not os.path.exists(manifest_path):
        print(f'Manifest file {manifest_path} does not exist')
        return -1

    # does the image directory exist?
    if not os.path.exists(image_directory):
        print(f'Directory {image_directory} does not exist')
        return -1
    
    # does the keypoints file exist?
    if not os.path.exists(keypoint_file):
        print(f'Could not find file {keypoint_file} with list of keypoints')
        return -1

    # is there a boundaries.txt file in there to split the images apart?
    if not os.path.exists(os.path.join(image_directory,'boundaries.txt')):
        print(f"boundaries.txt file does not exist. Place into {image_directory}")
        return -1

    # load the boundaries file into a dictionary
    with open(os.path.join(image_directory,'boundaries.txt')) as fid:
        boundaries = {line.split(':')[0]:bound_list2dict(eval(line.split(':')[1])) for line in fid.readlines()}


    # split images and labels into different views, save images and json
    splitter(manifest_path, boundaries, image_directory, keypoint_file)



# boundaries file list processor
def bound_list2dict(bound_list:List[int]):
    # view_names = ['North','South','East','West','Center']
    view_names = ['West', 'East', 'North', 'Center', 'South'] # same order as multiview_aws_preparation
    bound_dict = dict(zip(view_names,bound_list))

    return bound_dict




# split the images and the annotations
def splitter(manifest_path:str, boundaries:dict, image_dir:str, keypoint_file):
    
    # make a directory for the split files
    subdir_below = os.path.join(image_dir,'below')
    subdir_side = os.path.join(image_dir, 'side')
    if not os.path.exists(subdir_below):
        os.mkdir(subdir_below)  # create a subdirectory
    if not os.path.exists(subdir_side):
        os.mkdir(subdir_side)  # create a subdirectory


    # open the manifest
    with open(manifest_path, 'r') as manifest_fid:
        # # list of dicts for the labels
        label_list_below = []
        label_list_side = []

        # get the list of keypoints from the keypoints file
        with open(keypoint_file, 'r') as keypoint_fid:
            keypoint_labels = eval(keypoint_fid.readline())
            

        for line in manifest_fid.readlines():
            data = json.loads(line)
            
            # pull out image name, get the boundaries for this image
            image_name = os.path.split(data['source-ref'])[-1]
            boundary = boundaries[image_name]

            # do we have the image in the image_dir?
            if not os.path.exists(os.path.join(image_dir, image_name)):
                print(f'{image_name} not found in {image_dir}')
                continue
            
            # is the image found in the boundary list?
            if image_name not in boundaries.keys(): # skip this image if it's not in the boundary list
                print(f'{image_name} not found in boundaries.txt')
                continue

            # open the image
            image = cv2.imread(os.path.join(image_dir, image_name)) # open the image
        
            if 'annotatedResult' in data.keys():
                data = data['annotatedResult']

                # append data from all of the workers together
                entry_df = pd.DataFrame()
                for worker_entry in data['annotationsFromAllWorkers']:
                    # parse out the info
                    worker_df = pd.DataFrame.from_dict(eval(worker_entry['annotationData']['content'])['annotatedResult']['keypoints'])
                    worker_df['worker_id'] = worker_entry['workerId']
                    entry_df = pd.concat([entry_df, worker_df], ignore_index = True, join='outer')


                # create a sublist and run through each view.
                for bound_name, bound in boundary.items():
                    # which subimage?
                    subimage = os.path.splitext(image_name)[0] + '_' + bound_name + '.png'

                    if bound_name.lower() == 'center':
                        subdir = subdir_below
                    else:
                        subdir = subdir_side

                    # pop out the appropriate labels
                    bounded_df = entry_df.iloc[(entry_df.x.between(bound[1], bound[3]) & entry_df.y.between(bound[0], bound[2])).values]

                    # skip if there are less than 3 keypoints
                    # if len(bounded_df) == 0:
                    #     continue
                    if len(bounded_df) < 3:
                        continue

                    # split label into a dict
                    # this also assumeds the images will be padded so that center/below is 330x330, and side views to 240x210
                    label_dict,(width, height) = make_dict(bounded_df, bound, bound_name, keypoint_labels)
                    
                    # assemble the full entry
                    entry_dict = {
                                'image': os.path.join(subdir, subimage),
                                'height': height,
                                'width': width,
                                'ann_label': keypoint_labels,
                                'frame_id': os.path.join(subdir, subimage),
                                'ann_black': label_dict,
                    }


                    # split out the boundary portion of this image
                    im_crop = image[bound[0]:bound[2],bound[1]:bound[3]] # crop the image
                    im_crop = view_flipper(im_crop, bound_name).astype(np.uint8) # flip it. also introduce the padding

                    # save 'center' views separately from 'side' views
                    if bound_name.lower() == 'center':
                        temp_im = np.zeros((330,330,3), dtype=np.uint8)
                        temp_im[:im_crop.shape[0], :im_crop.shape[1],:] = im_crop
                        # cv2.imwrite(os.path.join(subdir_below, subimage), im_crop)
                        cv2.imwrite(os.path.join(subdir_below, subimage), temp_im)
                        label_list_below.append(entry_dict)
                    else:
                        temp_im = np.zeros((210, 240,3), dtype=np.uint8)
                        temp_im[:im_crop.shape[0], :im_crop.shape[1],:] = im_crop
                        # cv2.imwrite(os.path.join(subdir_side, subimage), im_crop)
                        cv2.imwrite(os.path.join(subdir_side, subimage), temp_im)
                        label_list_side.append(entry_dict)

            # if there aren't any labels in the output manifest...
            else:
                print(f'No data for {image_name} found in output manifest')


    # save the label lists
    with open(os.path.join(subdir_below, 'processed_keypoints.json'), 'w') as fid:
        # below_df.to_json(fid, orient='records')
        json.dump(label_list_below, fid)
    with open(os.path.join(subdir_side, 'processed_keypoints.json'), 'w') as fid:
        # side_df.to_json(fid, orient='records')
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
def make_dict(bounded_df:pd.DataFrame, bound, bound_name:str, keypoint_labels):

    # pull x and y into a list of lists -- grouped by worker
    label_df,(width, height) = label_view_parser(bounded_df, bound, bound_name)

    # creating a list of keypoints for each worker, with NaNs if the worker did not label this keypoint
    labels_x = label_df.groupby('worker_id').apply(lambda labels: [labels.loc[labels['label'] == keypoint]['x'].values[0]/width if labels['label'].eq(keypoint).any() else np.nan for keypoint in keypoint_labels]).values.tolist()
    labels_y = label_df.groupby('worker_id').apply(lambda labels: [labels.loc[labels['label'] == keypoint]['y'].values[0]/height if labels['label'].eq(keypoint).any() else np.nan for keypoint in keypoint_labels]).values.tolist()

    if len(labels_x[0]) < len(keypoint_labels):
        print('this is too short')

    # meds -- fill NaNs if the keypoints were not labeled in this frame
    med_x = label_df.groupby('label')['x'].median()/width
    med_x = [med_x[keypoint] if keypoint in med_x.index else np.nan for keypoint in keypoint_labels]
    med_y = label_df.groupby('label')['y'].median()/height
    med_y = [med_y[keypoint] if keypoint in med_y.index else np.nan for keypoint in keypoint_labels]

    # means
    mu_x = label_df.groupby('label')['x'].mean()/width
    mu_x = [mu_x[keypoint] if keypoint in mu_x.index else np.nan for keypoint in keypoint_labels]
    mu_y = label_df.groupby('label')['y'].mean()/height
    mu_y = [mu_y[keypoint] if keypoint in mu_y.index else np.nan for keypoint in keypoint_labels]

    # std
    std_x = label_df.groupby('label')['x'].std()/width
    std_x = [std_x[keypoint] if keypoint in std_x.index else np.nan for keypoint in keypoint_labels]
    std_y = label_df.groupby('label')['y'].std()/height
    std_y = [std_y[keypoint] if keypoint in std_y.index else np.nan for keypoint in keypoint_labels]

    # bounding box 
    # length for each side will be 1.6 * dist between extrema of medians
    Bx_off = abs(np.nanmax(med_x)-np.nanmin(med_x)) * 0.3 # offsets for bounding box
    By_off = abs(np.nanmax(med_y)-np.nanmin(med_y)) * 0.3 # offsets for bounding box
    B_xmin = max([0, np.nanmin(med_x) - Bx_off]) # don't go below 0
    B_xmax = min([1, np.nanmax(med_x) + Bx_off]) # don't go above width of image
    B_ymin = max([0, np.nanmin(med_y) - By_off])
    B_ymax = min([1, np.nanmax(med_y) + By_off])
    B_area = (B_xmax - B_xmin)*(B_ymax - B_ymin)*width*height


    label_dict = {'X': labels_x,
                  'Y': labels_y,
                  'bbox': [B_xmin, B_xmax, B_ymin, B_ymax],
                  'med': [med_y, med_x],
                  'mu': [mu_y, mu_x],
                  'std': [std_y, std_x],
                  'area': B_area
                  }
    
    return label_dict, (width, height)
    

# adjust the label points to account for rotating/reflecting the mirrors
def label_view_parser(bounded_df:pd.DataFrame, bound, bound_name:str):
    label_df = bounded_df.copy() # label_df == corrected dataframe

    if bound_name.lower() == 'center': # fit to boundaries
        label_df.x -= bound[1]
        label_df.y -= bound[0]
        # # width and height -- to swap if we rotate the image
        # height = bound[2] - bound[0]
        # width = bound[3] - bound[1]
        
        # padding it so that the images are consistent sizes per network
        height = 330
        width = 330
    
    if bound_name.lower() == 'north': # fit and flip vertically
        label_df.x -= bound[1]
        label_df.y = bound[2] - label_df.y
        # # width and height -- to swap if we rotate the image
        # height = bound[2] - bound[0]
        # width = bound[3] - bound[1]

        # padding it so that the images are consistent sizes per network
        height = 240
        width = 210

    if bound_name.lower() == 'south': # fit and flip horizontally
        label_df.x = bound[3] - label_df.x
        label_df.y -= bound[0]
        # # width and height -- to swap if we rotate the image
        # height = bound[2] - bound[0]
        # width = bound[3] - bound[1]
        # padding it so that the images are consistent sizes per network
        height = 240
        width = 210
    
    if bound_name.lower() == 'east': # transpose cropped x and y
        temp_y = label_df.x - bound[1] # future y == cropped (old) x
        label_df.x = label_df.y - bound[0] # x == cropped (old) y
        label_df.y = temp_y # should be a pd series
        # # width and height -- to swap if we rotate the image
        # height = bound[2] - bound[0]
        # width = bound[3] - bound[1]
        # padding it so that the images are consistent sizes per network
        height = 240
        width = 210

    if bound_name.lower() == 'west': # "anti-transpose"
        # I think this is x = maxY - y, y = maxX - x
        temp_y = bound[3]-label_df.x
        label_df.x = bound[2] - label_df.y
        label_df.y = temp_y
        # # width and height -- to swap if we rotate the image
        # height = bound[2] - bound[0]
        # width = bound[3] - bound[1]
        # padding it so that the images are consistent sizes per network
        height = 240
        width = 210


    return label_df, (width, height)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_path')
    parser.add_argument('image_directory', default='.')
    parser.add_argument('keypoint_file', default='.')

    args = parser.parse_args()

    output_manifest_processing(args.manifest_path, args.image_directory, args.keypoint_file)
