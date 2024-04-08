#! /usr/bin/env python

# multiview_utils
''' 
Utility functions for the multiview 3d setup.

This is primarily for splitting the images into multiple views using the 
boundaries text file from amazon or the calibration sqlite file, and
padding the images as needed to make sure that they work with whatever
model we're using.

'''

import os
import json
import numpy as np
import cv2
import sqlite3



# split image into different views based on boundaries.txt
def image_split_text(bound_file: str, image_dir:str):
    '''
    image_split_text

    arguments:
        - bound_file        text file from output_manifest_processing that defines boundaries

    '''

    pass



# split image into different views based on sql file
def video_split_sql(sql_path: str, video_path:str, output_dir:str = None, is_calib:bool = False):
    '''
    video_split_sql
        splits a multiview video into images of different views, including
        image corrections (flip, rotate etc) so that the floor of the side views
        is facing down. 

    arguments:
        - sql_path          sqlite file containing boundaries of views
        - video             path of video
        - output_dir        output directory. if None, creates new directory in same location as video
        - is_calib          is this a calibration video? if so, the sql query is a bit different [default = False]
    '''

    # check to make sure that we can access tables in the sql file
    if check_sql(sql_path) == -1:
        return -1

    # output directory if one wasn't provided
    if output_dir is None:
        output_dir = os.path.splitext(os.path.abspath(video_path))[0] + '_croppedViews'
    
    # create it if it doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # does the video exist
    if not os.path.exists(video_path):
        print(f'{video_path} does not exist')
        return -1


    # get the boundaries from sql
    boundaries = bound_puller(sql_path, video_path, is_calib)

    # open a video reader and writer for splitting
    vid_read = cv2.VideoCapture(video_path)

    # dict of videos writers -- one for each boundary
    vid_base = os.path.splitext(os.path.split(video_path)[-1])[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    vid_dict = {b_name:
                cv2.VideoWriter(os.path.join(output_dir, vid_base + '_' + b_name + '.mp4'), 
                                fourcc, 
                                vid_read.get(cv2.CAP_PROP_FPS),
                                (max(boundary[3]-boundary[1], boundary[2]-boundary[0]),
                                 min(boundary[3]-boundary[1], boundary[2]-boundary[0]))) 
                for b_name, boundary in boundaries.items()}
    # vid_dict = dict(zip([bound for bound in boundaries.keys()], ))
    
    # loop through the frames
    while True:

        # grab a frame
        ret,frame = vid_read.read()
        if not ret: # if we're out of frames hop out
            break

        # iterate through each boundary
        for b_name,bound in boundaries.items():
            temp_frame = frame[bound[0]:bound[2],bound[1]:bound[3],:].astype(np.uint8) # create a temp frame

            # manipulate it appropriately -- so far this will need to be hard coded
            temp_frame = view_flipper(temp_frame, b_name)

            # gamma correction
            temp_frame = ((temp_frame/255)**.6 * 255).astype(np.uint8)

            # save it
            vid_dict[b_name].write(temp_frame)

        
    # close the videos
    for b_name, b_video in vid_dict.items():
        b_video.release()

    


def bound_puller(sql_filename, vid_filename, is_calib:bool = False):
    '''
    get the boundaries of the file from the most recent video. 

    if is_calib == True, this video is a calibration video and just
    pull its bounding boxes directly
    '''
    if not os.path.exists(sql_filename):
        return -1
    
    conn = sqlite3.connect(sql_filename) # create connector
    cur = conn.cursor()

    vid_short = os.path.split(vid_filename)[-1]

    if not is_calib:
        # get the date of the video
        sql_query = "SELECT DATE(s.time), v.relative_path FROM session as s, videos as v WHERE v.relative_path LIKE ? AND s.rowid=v.session_id ;"
        response_video = np.array(cur.execute(sql_query,('%'+vid_short,)).fetchall())
        video_date = response_video[0,0].astype(np.datetime64) # convert to datetime for math
    
        # get all of the dates of the calibration videos
        sql_query = "SELECT DATE(c.date), c.boundary FROM calibration as c;"
        response_calibration = np.array(cur.execute(sql_query).fetchall())
        calib_date = response_calibration[:,0].astype(np.datetime64) # convert to datetime for math

        # find the associated calibration date
        date_diffs = (video_date-calib_date).astype(np.int8) # how long has passed since the calibration?
        calib_i = np.where(date_diffs < 0, np.inf, date_diffs).argmin() # find the most recent calibration (but not recorded after video)
        
        return eval(response_calibration[calib_i,1]) # return the bounding boxes as a dictionary
    
    elif is_calib:
        sql_query = "SELECT c.boundary FROM calibration as c WHERE c.relative_path LIKE ? ;"
        response_calibration = cur.execute(sql_query, ('%'+vid_short,)).fetchall()[0][0]

        return eval(response_calibration)



# flip views to account for the whole "mirror" thing
def view_flipper(image: np.array, view_name:str):
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



def check_sql(sql_path):
    '''
    connect to the sql make sure that we can actually get things from it
    '''

    if not os.path.exists(sql_path):
        print(f'Cannot find file {sql_path}')
        return -1

    # can we open the file?
    conn = sqlite3.connect(sql_path)
    cur = conn.cursor()

    cur.execute('PRAGMA table_list;')
    if len(cur.fetchall()) == 0:
        print(f'Did not find any tables in {sql_path}')
        return -1
    
    # close it all, return the file path
    cur.close()
    conn.close()
    return 1
