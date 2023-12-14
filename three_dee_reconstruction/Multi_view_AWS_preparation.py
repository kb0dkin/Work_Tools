# Multi-view AWS preparation
'''
With a list of videos, the script will pull out some random frames,
then create a manifest, directory with the frames clipped to only contain
the views, and the template html for the labeling tool

'''



from os import path, makedirs
import numpy as np
import cv2, glob, random, argparse, time
from typing import List
from pprint import pprint

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


def multi_view_preparation(output_dir:str = None, input_vids:List[str] = None, num_frames = 100):
    '''
    Put together everything needed for a multi-view AWS labeling setup.

    The script currently assumes we're working with the mirror-based food enclosure,
    so it splits the single view images (to remove all of the non-image parts) then 
    re-combines them to create a new series of images.

    For each run it will create:
    1. A series of images from the videos provided. Currently chosen randomly
    2. An AWS file manifest for all of the images.
    3. A template HTML file for the labeling tool.

    [optional] args:
    - 
    '''

    # create a new directory
    output_dir = create_subfolder(output_dir)
    
    # select the videos
    if (input_vids is None) or not any([path.exists(vid) for vid in input_vids]) :
        input_vids = select_vids(output_dir)
    
    # read videos, split them, then save them. 
    # might also be worth storing the 
    crop_and_splice(input_vids, output_dir, num_frames)



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


def bound_creator(vid:str):
    '''
    Displays a random frame in a video and has the user select different views
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
        center = tuple(np.floor(np.array(img.shape)/2)[::-1].astype(int))
        cv2.putText(img, bound_names[bound_i], center, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 3)
        
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
            center = tuple(np.floor(np.array(img.shape)/2)[::-1].astype(int))
            cv2.putText(img, bound_names[bound_i], center, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 3)


# split the image, put it back together
def crop_and_splice(video_paths, output_dir, num_frames):
    '''
    crop a video based on the bounds given, then splice them together into a single scene. 
    
    '''
    # for each video ....
    for i_video, video_path in enumerate(video_paths):
        # locations of views
        clear_bounds()
        bounds = bound_creator(video_path)
    
        # print(view_bounds)
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
        if not path.exists(output_dir):
            makedirs(output_dir)
        # if not path.exists(label_dir):
        #     makedirs(label_dir)

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
        vid_savename = path.join(output_dir,vid_basename + '_cropped.mp4') # to save the cropped file
        vid_write = cv2.VideoWriter(vid_savename, cv2.VideoWriter_fourcc(*'mp4v'), 50, (width, height))

        # get a list of frames to use -- random for now. I suppose in the future we could do K-means or PCA or something
        label_frames = random.choices(range(int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))), k = int(np.min([per_vid, frames_rem])))
        frames_rem -= per_vid # how many more do we need from future videos?

        # need to track the bounding boxes for the lambda function
        bound_fid = open(path.join(output_dir,'boundaries.txt'), 'w+')

        # loop through the frames
        i_frame = 0 # to keep track of whether we want to use this frame for labeling
        while True:
            # grab a frame
            ret,frame = vid_read.read()

            # if we're through the frames, leave the while loop
            if not ret:
                break

            # split the frame based on the crops, then put into the video
            fill_frame = np.zeros((height,width,3))
            for key in bounds.keys():
                bound = bounds[key]
                locn = target_corner[key]
                ws = width_subs[key]
                hs = height_subs[key]
                fill_frame[locn[0]:(locn[0]+hs),locn[1]:(locn[1]+ws),:] = frame[bound[0]:(bound[0]+hs), bound[1]:(bound[1]+ws),:]

            # write it to the output video            
            vid_write.write(fill_frame.astype(np.uint8)) # have to convert it to a uint

            # save it if it's a frame we want to label
            if i_frame in label_frames:
                im_filename = vid_basename + '_' + str(i_frame).zfill(8) + '.png'
                im_path = path.join(output_dir, im_filename) # filename with frame number
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
        bound_fid.close()

        




# arg parsing to run from the command line or just call straight
if __name__ == '__main__':
    multi_view_preparation()