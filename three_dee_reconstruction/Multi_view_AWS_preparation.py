# Multi-view AWS preparation
'''
With a list of videos, the script will pull out some random frames,
then create a manifest, directory with the frames clipped to only contain
the views, and the template html for the labeling tool

'''



from os import path, makedirs
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backend_bases import MouseButton
import numpy as np
import cv2, glob, random, argparse, time
from typing import List

# file explorer
from tkinter import Tk
from tkinter import filedialog as fd

ix = 0
iy = 0
drawing = 0
img = None
draw_img = None
bound_names = ['north','south','east','west','bottom']
bounds = {view:np.zeros((4,1)) for view in bound_names}
bound_i = 0


def multi_view_preparation(output_dir:str = None, input_vids:List[str] = None, num_images = 100):
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


    # locations of views
    view_bounds = bound_creator(input_vids)

    # # crop the images based on the bounds
    # # imgs_cropped = {img_num:img[view_bounds[img_num,[0,2]],view_bounds[img_num,[1,3]]] for img_num in range(view_bounds.shape[0])}

    # for num,imgs in imgs_cropped:
    #     while True:
    #         cv2.imshow(imgs_cropped)
    #         if (cv2.waitKey(1) & 0xFF) == 17:
    #             break







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


def bound_creator(input_vids:List[str]):
    '''
    Displays a random frame in a video and has the user select different views
    '''
    global img, draw_img, bounds, bound_i, bound_names # don't have a better way to pass them around right now
    
    # pull out a single image of a single video
    vid = random.choice(input_vids)
    cam = cv2.VideoCapture(vid)
    ret,frame = cam.read()
    # ret,img = cam.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        draw_img = img.copy() # create a copy that's used for cleaning up the dragged rectangles
        cv2.namedWindow('maskSelect', cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback('maskSelect', draw_mask)
        center = tuple(np.floor(np.array(img.shape)/2)[::-1].astype(int))
        print(center)
        # cv2.putText(img, bound_names[bound_i], np.floor(np.array(img.shape)/2).T.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,255,255))
        cv2.putText(img, bound_names[bound_i], center, cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 3)
        

        while bound_i < 5:
            cv2.imshow('maskSelect',img)
            k = cv2.waitKey(1) & 0xFF
            if k == 32: # escape on space key
                break
    
    cv2.destroyAllWindows()

    return bounds


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
        bounds[bound_names[bound_i]] = np.array([min(ix,x),min(iy,y),max(ix,x),max(iy,y)])
        bound_i += 1
        # cv2.putText(img, bound_names[bound_i], np.floor(np.array(img.shape)/2).T.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,255,255))
        cv2.putText(img, bound_names[bound_i], (1900,500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 3)


# arg parsing to run from the command line or just call straight
if __name__ == '__main__':
    multi_view_preparation()