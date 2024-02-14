# Multi-view AWS preparation
'''
take a calibration video to create a series of calibration
matrices and bounding boxes 

we'll then store the bounding box limits and the matrices in a 
sqlite database
'''



from os import path, makedirs
import os
import numpy as np
import cv2, glob, random, argparse, time
from typing import List
from pprint import pprint
from matplotlib import pyplot as plt
import sqlite3

# file explorer
from tkinter import Tk
from tkinter import filedialog as fd

ix = 0
iy = 0
drawing = 0
img = None
draw_img = None
bound_names = ['north','south','east','west','center']
bounds = {view:np.zeros((4,1)) for view in bound_names} # vertical, horizontal, height, width (treated as any other array)
bound_i = 0


def multiview_calibration_preparation(project_dir:str = None, input_vids:List[str] = None, num_frames = 100):
    '''
    Create bounding boxes and get calibration matrices using a calibration video.
    Then store the calibration in the base sqlite table


    The script currently assumes we're working with the mirror-based food enclosure,
    so it splits the single view images (to remove all of the non-image parts) 

    These bounding boxes and calibration matrices can then be applied to the mirror box
    recordings from the same day 

    '''

    # create a new directory
    sql_filename = connect_to_sql(project_dir, project_sql = None)
    if sql_filename == -1:
        print('Could not open sqlite file')
        return -1

    
    # select the videos
    if (input_vids is None) or not any([path.exists(vid) for vid in input_vids]) :
        input_vids = select_vids(project_dir)
    
    # # read videos, split them, calculate calibration matrices
    # # then store in the sqlite database
    # crop_and_splice(input_vids, project_dir, num_frames)

    # pull out boundaries for each video 
    for vid in input_vids:
        bound_creator(vid = vid)



def connect_to_sql(project_dir:str, project_sql:str):
    '''
    connect to the sql and return an open sqlite filename
    '''
    # replace the empty project director
    project_dir = '.' if project_dir is None else project_dir


    # open a UI file explorer if a file wasn't given
    if project_sql is None:
        root = Tk()
        project_sql = fd.askopenfilename(parent=root, title='Choose SQLite3 file', initialdir=project_dir)
        root.destroy()
    else:
        project_sql = path.join(project_dir,project_sql)



    # does the sql database exist? If not return an error
    if not path.exists(project_sql):
        return -1

    # can we open the file?
    conn = sqlite3.connect(project_sql)
    cur = conn.cursor()

    cur.execute('PRAGMA table_list;')
    if len(cur.fetchall()) == 0:
        return -1
    
    # close it all, return the file path
    cur.close()
    conn.close()
    return project_sql



def select_vids(project_dir):
    '''
    Choose videos to use for the labeling job
    '''
    root = Tk()
    input_vids = fd.askopenfilenames(parent=root, title='Calibration Videos', initialdir=path.split(project_dir)[0])
    root.destroy()
    return input_vids


# define the bounding boxes
def bound_creator(vid:str):
    '''
    Has the user outline bounding boxes for each view
    '''
    global img, draw_img, bounds, bound_i, bound_names # don't have a better way to pass them around right now
    
    # pull out a single image of a single video
    cam = cv2.VideoCapture(vid)
    ret,frame = cam.read()
    cam.release()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        draw_img = img.copy() # create a copy that's used for cleaning up the dragged rectangles

        # create a new window
        cv2.namedWindow('maskSelect', cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('maskSelect', draw_mask)

        # place instruction text in the center of the image
        text_size = cv2.getTextSize(bound_names[bound_i], cv2.FONT_HERSHEY_DUPLEX, 4, 3)[0]
        center = np.array(img.shape)[::-1] # flip x and y -- array wants y (vertical axis) first, cv wants x first
        center = (int((center[0]-text_size[0])/2),int((center[1]+text_size[1])/2))
        cv2.putText(img, bound_names[bound_i], center, cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 3)
        
        # limited to the number of bounds we have
        while bound_i < 5:
            cv2.imshow('maskSelect',img)
            k = cv2.waitKey(1) & 0xFF
            if k == 32: # escape on space key
                break
    
    cv2.destroyAllWindows()

    return bounds


def clear_bounds():
    '''
    resets the global variables bounds and bound_i

    This is so that we can have a different boundary setup per video
    '''
    global bounds, bound_i
    bound_i = 0
    bounds = {view:np.zeros((4,1)) for view in bound_names} # vertical, horizontal, height, width (treated as any other array)


def draw_mask(event, x, y, flags, params):
    ''' 
    openCV callback to segment and crop the views

    currently set to click-and-drag    
    '''
    global ix, iy, drawing, img, draw_img, bounds, bound_i, bound_names

    # on click create new rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        # draw_img=img.copy() # store current status of image
        drawing = True
        ix,iy = x,y

    # on drag expand the rectangle
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = draw_img.copy() # reset image so that we don't have overlapping rectangles
            cv2.rectangle(img, (ix,iy), (x,y), (255,255,255), 3)

    # on release draw the rectangle
    if event == cv2.EVENT_LBUTTONUP:
        # update the drawing to display properly
        draw_img = img.copy() # so we don't overwrite on mouse moves
        drawing = False
        cv2.rectangle(img, (ix,iy), (x,y), (255,255,255), 3)
        
        # update the list of view boundaries 
        bounds[bound_names[bound_i]] = np.array([min(iy,y),min(ix,x),max(iy,y),max(ix,x)])
        bound_i += 1

        # place instruction text in the center of the image
        if bound_i < 5:
            text_size = cv2.getTextSize(bound_names[bound_i], cv2.FONT_HERSHEY_DUPLEX, 4, 3)[0]
            center = np.array(img.shape)[::-1] # flip x and y -- array wants y (vertical axis) first, cv wants x first
            center = (int((center[0]-text_size[0])/2),int((center[1]+text_size[1])/2))
            cv2.putText(img, bound_names[bound_i], center, cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 3)


# loop through each video, take care of putting appropriate information into the database
def video_loop(video_paths, sqlite_path):
    
    sql_conn = sqlite3.connect(sqlite_path)
    sql_cur = sql_conn.cursor()



def crop_and_splice(video_paths, project_dir, num_frames):
    '''
    
    
    '''
    # need to track the bounding boxes for the lambda function
    bound_fid = open(path.join(project_dir,'boundaries.txt'), 'w+')

    
    # for each video ....
    for i_video, video_path in enumerate(video_paths):
        # locations of views
        clear_bounds()
        bounds = bound_creator(video_path)
    
        # get the widths and heighths of each view 
        # debating changing all of the xyxy to xywh...
        width_subs = {key:(bounds[key][3]-bounds[key][1]) for key in bounds.keys()}
        height_subs = {key:(bounds[key][2]-bounds[key][0]) for key in bounds.keys()}
    
        # the width will be the west and east image widths plus the largest width of north, center, and south
        width = width_subs['west'] + width_subs['east']
        width += max([width_subs['north'],width_subs['center'],width_subs['south']])

        # the height will be the west and east image heights plus the largest height of north, center, and south
        height = height_subs['north'] + height_subs['south']
        height += max([height_subs['west'],height_subs['center'],height_subs['east']])

        # locations of each view within the frame
        target_corner = dict()
        # west
        target_corner['west'] = [int((height - height_subs['west'])/2), 0]
        # east
        target_corner['east'] = [int((height - height_subs['east'])/2), width-width_subs['east']]
        # north
        target_corner['north'] = [0, int((width - width_subs['north'])/2)]
        # center
        target_corner['center'] = [int((height-height_subs['center'])/2), int((width-width_subs['center'])/2)]
        #south
        target_corner['south'] = [height - height_subs['south'], int((width - width_subs['south'])/2)]


        # the directory should already exist, but just in case...
        if not path.exists(project_dir):
            makedirs(project_dir)

        # how many label frames do we want per video?
        frames_rem = num_frames
        per_vid = int(np.ceil(num_frames/len(video_paths)))

        print(f'Cropping video {i_video+1} of {len(video_paths)}')

        # check to make sure the video exists. If not, print to console and skip
        if not path.exists(video_path):
            print(f'Couldn\'t find {video_path}. Continuing to next video.')

        # open a video reader and writer for the splitting
        vid_read = cv2.VideoCapture(video_path)
        vid_dirname, vid_filename = path.split(video_path) # get the storage location and video name
        vid_basename = path.splitext(vid_filename)[0] # for the cropped video and tagging frames
        vid_savename = path.join(project_dir,vid_basename + '_cropped.mp4') # to save the cropped file
        vid_write = cv2.VideoWriter(vid_savename, cv2.VideoWriter_fourcc(*'mp4v'), 50, (width, height))

        # get a list of frames to use -- random for now. I suppose in the future we could do K-means or PCA or something
        label_frames = random.choices(range(int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))), k = int(np.min([per_vid, frames_rem])))
        frames_rem -= per_vid # how many more do we need from future videos?


        # loop through the frames
        i_frame = 0 # to keep track of whether we want to use this frame for labeling
        while True:
            # grab a frame
            ret,frame = vid_read.read()

            # if we're through the frames, leave the while loop
            if not ret:
                break

            # skip everything but the label frames for the moment
            # if i_frame not in label_frames:
            #     i_frame += 1
            #     continue

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

            

            # write it to the output video            
            vid_write.write(fill_frame.astype(np.uint8)) # have to convert it to a uint

            # save it if it's a frame we want to label
            if i_frame in label_frames:
                # some additional instructions
                l1 = 'Label each keypoint:'
                l2 = '    - in at least 3 views'
                l3 = '    - only once per view'
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
                cv2.putText(fill_frame, b1, b1_origin, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
                cv2.putText(fill_frame, b2, b2_origin, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))



                # store it
                im_filename = vid_basename + '_' + str(i_frame).zfill(8) + '.png'
                im_path = path.join(project_dir, im_filename) # filename with frame number
                ret_im = cv2.imwrite(im_path,fill_frame) # write it
                if not ret_im:
                    frames_rem += 1 # still need to store another frame
                    print(f'Unable to save image {im_filename}') # let the user know

                boundary_list = [[target_corner[key][0],target_corner[key][1],target_corner[key][0]+height_subs[key], target_corner[key][1] + width_subs[key]] for key in target_corner.keys()]
                bound_fid.write(f'{im_filename}: {boundary_list}\n')


            # update the counter
            i_frame += 1

        # clean everything up for this loop
        vid_read.release()
        vid_write.release()
        
    # close boundary location file -- this is for all videos :)    
    bound_fid.close()

        




# arg parsing to run from the command line or just call straight
if __name__ == '__main__':
    multi_view_preparation()