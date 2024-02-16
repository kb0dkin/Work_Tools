# Multi-view AWS preparation
'''
Create a series of bounding boxes from the calibration videos.
We'll leave the calibration matrices to AniPose/DANNCE for now

we'll then store the bounding box limits in a sql database
'''



from os import path, makedirs
import os
import numpy as np
import cv2, glob, random, argparse, time
from typing import List
from pprint import pprint
from matplotlib import pyplot as plt
import sqlite3
import json # turning the dictionaries etc into something clean for sqlite

# file explorer
from tkinter import Tk
from tkinter import filedialog as fd

# argument parsing
import argparse

ix = 0
iy = 0
drawing = 0
img = None
draw_img = None
# bound_names = ['north','south','east','west','center']
# bounds = {view:np.zeros((4,1)) for view in bound_names} # vertical, horizontal, height, width (treated as any other array)
# bound_i = 0


class boundary():
    # class to keep track of boundaries during all of the cv2 callbacks
    def __init__(self, view_names):
        # initialize the whole thing
        self.view_names = view_names # names of each of the views
        self.bounds = {view:np.zeros((4,1)) for view in view_names} # dictionary of the boundary for each view
        self.i_bound = 0 # view counter -- for keeping track during the callback

    def print_bounds(self):
        # print out the bounding boxes in a somewhat readable way
        for key,value in self.bounds.items():
            print(f'{key}: {value}')

    def set_bounds(self, bounds:np.array):
        # set the bounding box, update the counter
        self.bounds[self.view_names[self.i_bound]] = bounds
        self.i_bound += 1

    def current_view(self):
        # just return the current view name
        return self.view_names[self.i_bound]
    
    def jsonify(self):
        # return a json version of the boundary dictionary
        return json.dumps(self.bounds)

class drag_drawing():
    # class to keep the images and x/y values for the dragging interface
    def __init__(self):
        self.ix = 0
        self.iy = 0
        self.drawing = 0
        self.img = None
        self.draw_img = None


class calibration_data():
    # store all calibration data for each of the views
    def __init__(self, view_names):
        self.camera_matrices = {view:[] for view in view_names}
        self.distortion_coefficients = {view:[] for view in view_names}
        self.rotation_vectors = {view:[] for view in view_names}
        self.translation_vectors = {view:[] for view in view_names}




# class to keep track of all of the chAruco board stuff
class chAruco_board():

    def __init__(self):
        # initialize.
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6x6_250)
        board = cv2.aruco.CharucoBoard_create(6,6,1,.8,aruco_dict)
        imboard = board.draw((2000,2000))

        # save in the structure
        self.aruco_dict = aruco_dict
        self.board = board
        self.imboard = imboard




def multiview_calibration_preparation(project_dir:str = '.', input_vids:List[str] = None, sql_path:str = None,
                                      view_names:List[str] = ['North','South','East','West','Center']):
    '''
    Create bounding boxes and get calibration matrices using a calibration video.
    Then store the calibration in the base sqlite table


    The script currently assumes we're working with the mirror-based food enclosure,
    so it splits the single view images (to remove all of the non-image parts) 

    These bounding boxes and calibration matrices can then be applied to the mirror box
    recordings from the same day 

    '''

    # create a new directory
    sql_path = connect_to_sql(project_dir, sql_path)
    if sql_path == -1:
        print('Could not open sqlite file')
        return -1

    
    # select the videos
    if (input_vids is None) or not any([path.exists(vid) for vid in input_vids]) :
        input_vids = select_vids(path.split(sql_path)[0])
    

    for vid in input_vids:
        # pull out boundaries for each video 
        vid_bounds = bound_creator(vid = vid, view_names=view_names)
        
        # find the calibration matrices
        # crop_and_calc(vid = vid, bounds = vid_bounds)
       
        # write to sql
        sql_write(sql_path, vid_bounds)



def connect_to_sql(project_dir, sql_path):
    '''
    connect to the sql and return an sqlite filepath
    '''


    if sql_path is not None and not path.exists(sql_path):
        sql_path = path.join(project_dir, sql_path)

    # open a UI file explorer if a file wasn't given
    if sql_path is None or not path.exists(sql_path):
        root = Tk()
        sql_path = fd.askopenfilename(parent=root, title='Choose SQLite3 file', initialdir=project_dir)
        root.destroy()


    # can we open the file?
    conn = sqlite3.connect(sql_path)
    cur = conn.cursor()

    cur.execute('PRAGMA table_list;')
    if len(cur.fetchall()) == 0:
        return -1
    
    # close it all, return the file path
    cur.close()
    conn.close()
    return sql_path



def select_vids(project_dir):
    '''
    Choose videos to use for the labeling job
    '''
    root = Tk()
    input_vids = fd.askopenfilenames(parent=root, title='Calibration Videos', initialdir=path.split(project_dir)[0])
    root.destroy()
    return input_vids


# define the bounding boxes
def bound_creator(vid:str, view_names):
    '''
    Has the user outline bounding boxes for each view
    '''

    # create a new instance of a boundary
    bound_instance = boundary(view_names=view_names)

    # and a drag_drawing parameter holder
    drag_instance = drag_drawing()

    # combine into a params dictionary
    params_dict = {'bound_instance':bound_instance, 'drag_instance':drag_instance}

    # pull out a single image of a single video
    cam = cv2.VideoCapture(vid)
    ret,frame = cam.read() # read in a frame
    cam.release() # release the connector
    if ret:
        drag_instance.img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        drag_instance.draw_img = drag_instance.img.copy() # create a copy that's used for cleaning up the dragged rectangles

        # create a new window
        cv2.namedWindow('maskSelect', cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

        # place instruction text in the center of the image
        text_size = cv2.getTextSize(bound_instance.current_view(), cv2.FONT_HERSHEY_DUPLEX, 4, 3)[0]
        center = np.array(drag_instance.img.shape)[::-1] # flip x and y -- array wants y (vertical axis) first, cv wants x first
        center = (int((center[0]-text_size[0])/2),int((center[1]+text_size[1])/2))
        cv2.putText(drag_instance.img, bound_instance.current_view(), center, cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 3)

        # connect a callback to draw the masks
        cv2.setMouseCallback('maskSelect', draw_mask, params_dict)


        # limited to the number of bounds we have
        while bound_instance.i_bound < 5:
            cv2.imshow('maskSelect',drag_instance.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 32: # escape on space key
                break
    
    cv2.destroyAllWindows()

    return bound_instance


def draw_mask(event, x, y, flags, params):
    ''' 
    openCV callback to segment and crop the views

    currently set to click-and-drag    
    '''
    # global ix, iy, drawing, img, draw_img
    # global ix, iy, drawing, img, draw_img, bounds, bound_i, bound_names

    # parse out the params dictionary
    bound_instance = params['bound_instance']
    drag_instance = params['drag_instance']

    # on click create new rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_instance.drawing = True
        drag_instance.ix,drag_instance.iy = x,y

    # on drag expand the rectangle
    if event == cv2.EVENT_MOUSEMOVE:
        if drag_instance.drawing == True:
            drag_instance.img = drag_instance.draw_img.copy() # reset image so that we don't have overlapping rectangles
            cv2.rectangle(drag_instance.img, (drag_instance.ix,drag_instance.iy), (x,y), (255,255,255), 3)

    # on release draw the rectangle
    if event == cv2.EVENT_LBUTTONUP:
        # update the drawing to display properly
        drag_instance.draw_img = drag_instance.img.copy() # so we don't overwrite on mouse moves
        drag_instance.drawing = False
        cv2.rectangle(drag_instance.img, (drag_instance.ix,drag_instance.iy), (x,y), (255,255,255), 3)
        
        # update the list of view boundaries 
        bound_instance.set_bounds(np.array([min(drag_instance.iy,y),min(drag_instance.ix,x),max(drag_instance.iy,y),max(drag_instance.ix,x)]))

        # place instruction text in the center of the image
        if bound_instance.i_bound < 5:
            text_size = cv2.getTextSize(bound_instance.current_view(), cv2.FONT_HERSHEY_DUPLEX, 4, 3)[0]
            center = np.array(drag_instance.img.shape)[::-1] # flip x and y -- array wants y (vertical axis) first, cv wants x first
            center = (int((center[0]-text_size[0])/2),int((center[1]+text_size[1])/2))
            cv2.putText(drag_instance.img, bound_instance.current_view(), center, cv2.FONT_HERSHEY_DUPLEX, 4, (255,255,255), 3)


# write view bounds and matrices for each view to the sqlite database
def sql_write(sqlite_path, view_bounds):
    # connect to the sqlite db    
    sql_conn = sqlite3.connect(sqlite_path)
    sql_cur = sql_conn.cursor()

    

    # format the sql insertion
    sql_query = f""


# main function to handle the calibration matrices etc
def crop_and_calc(video, bounds, calibration):

    # instance of the chAruco board
    charuco_instance = chAruco_board()

    # dictionary to keep the frames as we crop them
    images = {view:[] for view in bounds.view_names}

    # open the video, pull in images and crop
    cam = cv2.VideoCapture(video)
    ret,frame = cam.read() # read in a frame
    while ret: # keep reading until we run out of frames
        





    




# find  location of the chessboard in each image
def read_chessboards(images, chAruco_instance):
    allCorners = [] # all chessboard corners
    allIds = [] # all Aruco IDs
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001) # subpixel corner criteria

    for im in images:
        frame = cv2.imread(im,0) # read it in grayscale
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, chAruco_instance['aruco_dict']) # from definition above

        if len(corners) > 0:
            # sub pixel detection, apparently. assuming linear interpolation?
            for corner in corners:
                cv2.cornerSubPix(frame, corner, winSize=(3,3), zeroZone=(-1,-1), criteria=criteria) # not sure about this -- no output...
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, chAruco_instance['board'])
            if res2[1] is not None and res2[2] is not None and len(res2[1])> 3:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        
    
    imsize = frame.shape
    return allCorners, allIds, imsize



def calibrate_camera(allCorners,allIds,imsize, chAruco_instance):
    """
    Calibrates the camera using the detected corners.
    """

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=chAruco_instance['board'],
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors



def crop_and_splice(video_paths, project_dir, num_frames):
    '''
    
    
    '''
    # need to track the bounding boxes for the lambda function
    bound_fid = open(path.join(project_dir,'boundaries.txt'), 'w+')

    
    # for each video ....
    for i_video, video_path in enumerate(video_paths):
        # locations of views
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
    '''
    Create bounding boxes and calibration matrices from a calibration video.
    '''



    # parsing them args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--sql', help='SQLite3 file name or path', default=None)
    parser.add_argument('--directory',help='Project Base Directory', default=None)
    parser.add_argument('-v','--video',help='Calibration video', default=None)

    args = parser.parse_args()

    multiview_calibration_preparation(project_dir=args.directory, input_vids=args.video, sql_path=args.sql)