# multiview_aws_prep
#
# Take a bounding box setup from the sql file and spit out cropped/spliced frames.
#
# 

import numpy as np
import cv2, glob, random, argparse, time
from typing import List

# file explorer
from tkinter import Tk
from tkinter import filedialog as fd


def multiview_aws_prep(input_vids:str = None, sql_file:str = None, output_dir:str = None):
    '''
    Put together everything needed for a multi-view AWS labeling setup.

    The script currently assumes we're working with the mirror-based food enclosure,
    so it splits the single view images (to remove all of the non-image parts) then 
    re-combines them to create a new series of images.

    For each it will pull in the boundaries in the associated sqlite directory

    [optional] args:
    - output_dir    :   where do we store the frames? Opens a GUI dialog if not specified
    - input_vids    :   videos to clip. Opens a GUI dialog if not specified
    - sql_file      :   sqlite file that contains the boundaries
    - calib_file    :   calibration file name. Chooses the date closest to 
    - 
    '''
