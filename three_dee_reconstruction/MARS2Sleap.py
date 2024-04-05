#! /usr/bin/env python

# MARS2Sleap
'''
Converts a MARS processed_keypoints.json into a .slp project
'''

import sleap
import json
import os
import argparse


def MARS2Sleap(keypoints_json:str, data_path:str, with_images:bool = False):
    '''
    Converts a MARS processed_keypoints.json into a .slp project

    Arguments:
        - keypoints_json    path to the keypoints json
        - data_path         directory with image files, where to save the .slp project
        - with_images       Should we save the images in the .slp?

    '''

    if keypoints_json is None or data_path is None:
        print('Please give the path to the json and the images')
        return -1
    

    # load in the annotations
    with open(keypoints_json, 'r') as fid:
        anns = json.load(fid)
    nodes = list(set([label for ann in anns for label in ann['ann_label']]))

    # initialize the skeleton
    skeleton = sleap.Skeleton()
    skeleton.add_nodes(nodes)


    # create a sleap video from the filenames.
    frame_names = [data_path + os.path.split(ann['image'])[-1] for ann in anns]
    video = sleap.Video.from_image_filenames(filenames=frame_names)


    # create a labeled frame iterator.
    # right now it's just looking for ann_black, but we should change this to multiple
    # instances in the future
    frames = []
    # iterate through the images
    for ann in anns:

        height = ann['height']
        width = ann['width']

        # get the image name
        img_name = os.path.split(ann['image'])[-1]
        img_id = video.get_idx_from_filename(data_path+img_name)
    
        instances = []
        keypoint_dict = {ann_label:sleap.instance.Point(x=ann['ann_black']['X'][0][ii]*width,y=ann['ann_black']['Y'][0][ii]*height)
                        for ii,ann_label in enumerate(ann['ann_label'])}
        instance = sleap.Instance(skeleton=skeleton, points=keypoint_dict)
        instances.append(instance)

        labeled_frame = sleap.LabeledFrame(video=video, frame_idx=img_id, instances=instances)

        # put the instances into the labeled frame list
        frames.append(labeled_frame)

    # put the frames into a sleap labels list
    output_labels = sleap.Labels(frames)

    # save it all
    output_labels.save(os.path.join(data_path,'processed_keypoints.slp'),with_images=with_images)






# main -- to call as a script
if __name__ == '__main__':
    '''
    Converts a MARS processed_keypoints.json into a .slp project
    '''


    parser = argparse.ArgumentParser(description='Converts a MARS processed_keypoints.json into a .slp project')
    parser.add_argument('keypoints_json', default=None, help='path to processed_keypoints.json')
    parser.add_argument('data_path', default=None, help='directory of images, where output .slp will be stored')
    parser.add_argument('-i', action='store_true', dest='with_images', help='Save images in .slp file?')

    args = parser.parse_args()
    MARS2Sleap(args.keypoints_json, args.data_path, args.with_images)