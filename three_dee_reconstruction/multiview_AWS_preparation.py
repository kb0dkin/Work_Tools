# multiview AWS preparation
'''
Crops and stores frames using bounding boxes from a provided sqlite directory

Stores the frames and a list of view bounding boxes (in the cropped coordinate frame)
so that we can get them labeled by AWS.
'''



from os import path, makedirs
import numpy as np
import cv2, random, argparse
from typing import List
from pprint import pprint
from matplotlib import pyplot as plt

# file explorer
from tkinter import Tk
from tkinter import filedialog as fd

# connecting to sqlite
import sqlite3


def multiview_aws_preparation(output_dir:str = None, input_vids:List[str] = None, sql_file:str = None, num_frames:int = 100):
    '''
    Put together everything needed for a multi-view AWS labeling setup.

    The script currently assumes we're working with the mirror-based food enclosure,
    so it splits the single view images (to remove all of the non-image parts) then 
    re-combines them to create a new series of images.

    For each it will pull in the boundaries in the associated sqlite directory

    [optional] args:
    - output_dir    :   where do we store the frames? Opens a GUI dialog if not specified
    - input_vids    :   videos to clip. Opens a GUI dialog if not specified
    - sql_file      :   sqlite file that contains the boundaries. Opens GUI if not found
    - calib_file    :   calibration file name. Default chooses the most recent calib video
    - num_frames    :   number of frames to output. Default 100
    '''

    # create a new directory
    output_dir = create_subfolder(output_dir)
    
    # select the videos
    if (input_vids is None) or not any([path.exists(vid) for vid in input_vids]) :
        input_vids = select_vids(output_dir)

    # select the sqlite file
    if sql_file is None:
        root = Tk()
        sql_file = fd.askopenfilename(parent=root, initialdir=output_dir, title='Select sqlite3 file')
        root.destroy()
        
    # read out frames, crop and store them
    crop_and_splice(input_vids, output_dir, num_frames, sql_file)


def create_subfolder(output_dir):
    '''
    create a new subdirectory for the images
    '''
    # use the file explorer if we weren't given one from the command line
    if output_dir is None:
        root = Tk()
        source_path = fd.askdirectory(parent=root, title='Select the Parent Directory')
        output_dir = path.join(source_path,'AWS_labeling_setup')
        root.destroy()
    
    # create a new directory
    if not path.exists(output_dir):
        makedirs(output_dir)

    return output_dir


def select_vids(output_dir):
    '''
    Choose videos to use for the labeling job
    '''
    root = Tk()
    input_vids = fd.askopenfilenames(parent=root, title='Videos for Labeling', initialdir=path.split(output_dir)[0])
    root.destroy()
    return input_vids


def bound_puller(sql_filename, vid_filename):
    '''
    get the boundaries of the file from the most recent video
    '''
    if not path.exists(sql_filename):
        return -1
    
    conn = sqlite3.connect(sql_filename) # create connector
    cur = conn.cursor()

    vid_short = path.split(vid_filename)[-1]

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


# split the image, put it back together
def crop_and_splice(video_paths:List[str], output_dir:str, num_frames:int, sql_filename:str):
    '''
    crop a video based on the bounds given, then splice them together into a single scene. 
    
    '''
    # the "target" view boundaries -- for the crop and splice version of the frame
    bound_fid = open(path.join(output_dir,'boundaries.txt'), 'w+')

    # how many label frames do we want per video?
    frames_rem = num_frames
    per_vid = int(np.ceil(num_frames/len(video_paths)))

    # for each video ....
    for i_video, video_path in enumerate(video_paths):
        # pull bounding boxes from the sqlite database
        bounds = bound_puller(sql_filename, video_path)

        # get the widths and heights of each view 
        # debating changing all of the xyxy to xywh...
        width_subs = {key:(bounds[key][3]-bounds[key][1]) for key in bounds.keys()}
        height_subs = {key:(bounds[key][2]-bounds[key][0]) for key in bounds.keys()}
    
        # the width will be the West and East image widths plus the largest width of North, Center, and South
        width = width_subs['West'] + width_subs['East']
        width += max([width_subs['North'],width_subs['Center'],width_subs['South']])

        # the height will be the West and East image heights plus the largest height of North, Center, and South
        height = height_subs['North'] + height_subs['South']
        height += max([height_subs['West'],height_subs['Center'],height_subs['East']])

        # locations of each view within the frame
        target_corner = dict()
        # West
        target_corner['West'] = [int((height - height_subs['West'])/2), 0]
        # East
        target_corner['East'] = [int((height - height_subs['East'])/2), width-width_subs['East']]
        # North
        target_corner['North'] = [0, int((width - width_subs['North'])/2)]
        # Center
        target_corner['Center'] = [int((height-height_subs['Center'])/2), int((width-width_subs['Center'])/2)]
        #South
        target_corner['South'] = [height - height_subs['South'], int((width - width_subs['South'])/2)]


        # the directory should already exist, but just in case...
        if not path.exists(output_dir):
            makedirs(output_dir)


        print(f'Cropping video {i_video+1} of {len(video_paths)}')

        # check to make sure the video exists. If not, print to console and skip
        if not path.exists(video_path):
            print(f'Couldn\'t find {video_path}. Continuing to next video.')

        # open a video reader and writer for the splitting
        vid_read = cv2.VideoCapture(video_path)
        vid_dirname, vid_filename = path.split(video_path) # get the storage location and video name
        vid_basename = path.splitext(vid_filename)[0] # for the cropped video and tagging frames

        # get a list of frames to use -- random for now. I suppose in the future we could do K-means or PCA or something
        label_frames = random.choices(range(int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))), k = int(np.min([per_vid, frames_rem])))
        frames_rem -= per_vid # how many more do we need from future videos?


        # loop through the frames
        for i_frame in label_frames:
            # move the cursor to the desired frame, then grab a frame
            vid_read.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
            ret,frame = vid_read.read()

            # if we couldn't open the frame, skip it and add one to the "frames remaining" counter
            if not ret:
                frames_rem += 1 # still need to store another frame
                continue

            # split the frame based on the crops, then put into the video
            fill_frame = np.zeros((height,width,3))
            for key in bounds.keys():
                bound = bounds[key]
                locn = target_corner[key]
                ws = width_subs[key]
                hs = height_subs[key]
                
                # gamma correction
                frame_temp = frame[bound[0]:(bound[0]+hs), bound[1]:(bound[1]+ws),:]
                frame_temp = ((frame_temp/255)**.6 * 255).astype(np.uint8)

                # stick the frame in there
                fill_frame[locn[0]:(locn[0]+hs),locn[1]:(locn[1]+ws),:] = frame_temp


            # some additional instructions
            l1 = 'Label each keypoint:'
            l2 = '    - in as many views'
            l3 = '         as possible'
            l4 = '    - only once per view'
            b1 = 'Refer to instructions'
            b2 = 'for view layout'
            inst_scale = 0.5
            top_size = cv2.getTextSize(l1, cv2.FONT_HERSHEY_SIMPLEX, inst_scale,1)[0]
            b1_size = cv2.getTextSize(b1, cv2.FONT_HERSHEY_SIMPLEX, inst_scale,1)[0]
            b2_size = cv2.getTextSize(b2, cv2.FONT_HERSHEY_SIMPLEX, inst_scale,1)[0]
            b1_origin = (int(fill_frame.shape[1]-(b1_size[0]+5)),int(fill_frame.shape[0] - 2*b1_size[1])) 
            b2_origin = (int(fill_frame.shape[1]-(b2_size[0]+5)),int(fill_frame.shape[0] - 0.5*b2_size[1]))
            cv2.putText(fill_frame, l1, (5,int(1.5*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
            cv2.putText(fill_frame, l2, (5,int(3*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
            cv2.putText(fill_frame, l3, (5,int(4.5*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
            cv2.putText(fill_frame, l4, (5,int(6*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
            cv2.putText(fill_frame, b1, b1_origin, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
            cv2.putText(fill_frame, b2, b2_origin, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))



                # store it
            im_filename = vid_basename + '_' + str(i_frame).zfill(8) + '.png'
            im_path = path.join(output_dir, im_filename) # filename with frame number
            ret_im = cv2.imwrite(im_path,fill_frame) # write it
            if not ret_im:
                frames_rem += 1 # still need to store another frame
                print(f'Unable to save image {im_filename}') # let the user know
                continue

            boundary_list = [[target_corner[key][0],target_corner[key][1],target_corner[key][0]+height_subs[key], target_corner[key][1] + width_subs[key]] for key in target_corner.keys()]
            bound_fid.write(f'{im_filename}: {boundary_list}\n')



        # clean everything up for this loop
        vid_read.release()
        
    # close boundary location file -- this is for all videos :)    
    bound_fid.close()


# arg parsing to run from the command line
if __name__ == '__main__':
    '''
    Extracts frames for the multiview recording setup for AWS labeling, 
    then crops them from the associated calibration video.
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--outputdir', default=None, help='Output directory for frames pulled for labeling')
    parser.add_argument('-s', '--sqlfile', default=None, help='SQLite3 file name.')
    parser.add_argument('-n', '--numframes', default=100, help='Number of frames to extract', type=int)
    # parser.add_argument('--inputvids', default=None, action='append', help='Videos for labeling. One flag per video')
    parser.add_argument('--inputvids', default=None, nargs='*', help='Videos for labeling. One flag per video')

    args = parser.parse_args()

    multiview_aws_preparation(output_dir=args.outputdir, input_vids=args.inputvids, num_frames=args.numframes, sql_file=args.sqlfile)
