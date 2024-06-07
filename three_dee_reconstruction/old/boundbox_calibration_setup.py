# Run this on a calibration video, then use the created bits for behavior videos.
# 
# Creates a series of opencv calibration matrices for the multi-view mouse house
# It also allows the user to select masks for each camera view.
# It then stores the locations of these bounding boxes and calibration matrices
# into a sqlite database
# 
# First, import all necessary libraries
# 

# %%
import cv2, glob, random
from os import path, mkdir
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.backend_bases import MouseButton
import numpy as np
import time
%matplotlib ipympl


# %% [markdown]
# ## Create the ChAruco board
# this is currently setup for a 6x6 board with a Aruco square with a vertex length of .8*chessboard tile length. It will save the board as a tiff in the desired "workdir"

# %%
workdir = '~/' # just put it in home for now
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard_create(6,6,1,.8,aruco_dict)
imboard = board.draw((2000,2000))
cv2.imwrite(workdir+"ChAruco.tiff",imboard)

fig,ax = plt.subplots()
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation='nearest')
ax.axis('off')

# %% [markdown]
# ## Create the masks
# These masks are to split each image into the different views -- since we'll be working with mirrors, and the location of each view might change a little in each recording if the camera gets bumped or something
# 

# %%
base_path = '/home/klb807/Documents/three_dee_calib_tests/'

images = glob.glob(base_path+'*.tiff')

num_images = 1 # how many images do we want to use for cropping?
num_masks = 5 # how many crops (masks) do we want per image?


class mask():
    def __init__(self, num_masks, image_filenames):
        self.num_images = len(image_filenames)
        self.num_masks = num_masks
        self.image_filenames = image_filenames

        self.image_index = 0
        self.mask_index = 0
        self.point_index = 0

        self.mask = np.ndarray((num_images, num_masks, 4, 2))
        self.mask[:] = np.nan
                    
                        
def on_click(event, mask):
    if (event.button is MouseButton.LEFT):
        mask.mask[mask.image_index, mask.mask_index, mask.point_index, 0] = event.xdata # x location
        mask.mask[mask.image_index, mask.mask_index, mask.point_index, 1] = event.ydata # y location
        ax.plot(event.xdata, event.ydata, color='forestgreen', marker='*')
        # print(f"x: {event.x}, x_data: {event.xdata}")


        # update indices for appropriate mask, and show the polygon
        mask.point_index = np.mod(mask.point_index + 1, 4) # assume 4 points per mask
        if mask.point_index == 0: # if we've done all of the points in a mask
            ax.add_patch(Polygon(mask.mask[mask.image_index, mask.mask_index, :,:], color='forestgreen', alpha=.25))
            mask.mask_index = np.mod(mask.mask_index + 1, mask.num_masks) # update mask index
            if mask.mask_index == 0: # if we've gone through the masks for an image
                mask.image_index += 1
                if mask.image_index == mask.num_images: # if we've shown all of the images
                    plt.disconnect(binding_id)
                    print(f"callback disconnected")
                    plt.close(fig)
                        
                else: # if we still have more images to show, display the next one
                    # im = PIL.Image(image_filenames[image_index])
                    ax.clear()
                    im = cv2.imread(mask.image_filenames[mask.image_index])
                    ax.imshow(im, cmap=mpl.cm.gray)

    elif event.button is MouseButton.RIGHT:
        plt.disconnect(binding_id)
        print(f"callback disconnected")




fig,ax = plt.subplots()

image_filenames = random.choices(images, k=num_images)
im = cv2.imread(image_filenames[0])
ax.imshow(im)
mask = mask(num_masks=num_masks, image_filenames=image_filenames)
binding_id = plt.connect('button_press_event', lambda x: on_click(x, mask))
# plt.show()


# %% [markdown]
# ## Find the average for each mask
# Find an average mask for each location, then crop all of the images in the directory according to that cropping and work from there.
# 
# We can't guarantee the order of tagged points, so we'll basically find the closest point for each image and average those. 
# Final array should be (num_masks)x4x2

# %%
mean_masks = np.ndarray((num_masks, 4, 2))

if num_images == 1:
    mean_masks = mask.mask[0,:,:,:]
else:
    # for each mask for the first image
    for mask_ind in range(num_masks):
        curr_mask = mask.mask[0,mask_ind,:,:] # get the current mask

        for i_point in range(4): # for each point in the current mask
            curr_point = curr_mask[i_point, :]
            points = np.ndarray((num_images,2)) # find the point from each image that's closest to the current point
            points[0,:] = curr_mask[i_point, :] # setup the first point
            for image_ind in range(1,2):
                all_points_from_image = np.reshape(mask.mask[image_ind, :, :, :], (num_masks*4, 2))
                point_ind = np.argmin(np.sum((all_points_from_image - curr_point)**2, axis=1))
                points[image_ind, :] = all_points_from_image[point_ind,:]

            mean_masks[mask_ind,i_point,:] = np.mean(points, axis=0)


# %% [markdown]
# ### Split images and save
# For this we'll 
# 
# 1. Create a new subdirectory for each mask
# 2. Split each image by each mask and save into the associated directory
# 3. Display a couple of example images

# %%
# make new directories
subdir_names = ['mask_'+str(ii) for ii in range(num_masks)]
for subdir in subdir_names:
    if not path.isdir(path.join(base_path, subdir)):
        mkdir(path.join(base_path,subdir))

for i_filename, filename in enumerate(images):
    image = cv2.imread(filename)
    _, base_filename = path.split(filename)
    for i_mask in range(num_masks):
        crop = image[int(min(mean_masks[i_mask,:,1])):int(max(mean_masks[i_mask,:,1])), # x values
                     int(min(mean_masks[i_mask,:,0])):int(max(mean_masks[i_mask,:,0])), :] # y values
        result = cv2.imwrite(path.join(base_path,subdir_names[i_mask],base_filename), crop)
        if result == False:
            print(f"Could not write image {path.join(base_path,subdir_names[i_mask],base_filename)}")


# %% [markdown]
# ## now get the distortion array etc for each view
# opencv fortunately has functions to do this for us
# 
# we'll start by creating a function to parse the ChAruco board. This comes from tutorials from both Aruco2 and OpenCV

# %%
def read_chessboards(images):
    allCorners = [] # all chessboard corners
    allIds = [] # all Aruco IDs
    decimator = 0 # for sub-pixel estimation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001) # subpixel corner criteria

    for im in images:
        frame = cv2.imread(im,0) # read it in grayscale
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict) # from definition above

        if len(corners) > 0:
            # sub pixel detection, apparently. assuming linear interpolation?
            for corner in corners:
                cv2.cornerSubPix(frame, corner, winSize=(3,3), zeroZone=(-1,-1), criteria=criteria) # not sure about this -- no output...
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])> 3 and decimator%1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        
        decimator += 1
    
    imsize = frame.shape
    return allCorners, allIds, imsize



# %% [markdown]
# and define the calibration function -- this will return the camera matrix, distortion coefficients, rotation vectors etc    

# %%
def calibrate_camera(allCorners,allIds,imsize):
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
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

# %% [markdown]
# now run each of them for each view

# %%
images = glob.glob(path.join(base_path,'mask_0','*.tiff'))

allCorners, allIds, imsize = read_chessboards(images)
ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize)


# %% [markdown]
# Now we're going to draw the 3d axis from the chessboard corner in each image

# %%
def draw_axis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)
    return img

# %%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001) # subpixel corner criteria
objp = np.zeros((6*6, 3), np.float32)
objp[:,:2] = np.mgrid[1:2:6, 1:2:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)

# %%
for image in glob.glob(path.join(base_path, 'mask_0', '*.tiff')):
    img = cv2.imread(image, 0)
    # ret, corners = cv2.findChessboardCorners(img, (6,6), None)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict) # from definition above
    # convert from list to arrays, then reshape into Nx2 (N=number of squares x 4 points per square), 
    # then sort by id
    corner_array = np.reshape(np.array(corners),(-1,2))
    id_reorder = 

    # if ret == True:
    if len(corners) > 0:
        corner_array = np.array(corners)
        if [0] in ids:
            corners2 = cv2.cornerSubPix(img, corner_array[np.where(ids==0)].squeeze(), (11,11), (-1,-1), criteria)

            # find the rotation and translation vectors)
            # ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            # project 3d points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            # img = draw_axis(img, corners2, imgpts)
            img = draw_axis(img, corners2, imgpts)
            img_name = image[:4]+'_axis.png'
            ret = cv2.imwrite(img_name, img)

        if ret == False:
            print(f"could not save image {img_name}")
    
    else:
        print(f"No board found for {image}")

cv2.destroyAllWindows()

# %% [markdown]
# ## Create an html template file to load on to Amazon
# specifically for the multi-view situation

# %%
from generate_AWS_multiview_template import generate_AWS_template

# %%
generate_AWS_template(['Nose','Right Ear','Left Ear','Throat','Body Center','Right Hip','Left Hip','Tail Base'])

# %%
import os
os.path.abspath('/home/klb807/')

# %%
import Multi_view_AWS_preparation

# %%
Multi_view_AWS_preparation.multi_view_preparation(output_dir='C:\\Users\\17204\\Documents\\Data\\Cropped', num_frames=1)

# %%
import numpy as np
import cv2 as cv
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv.circle(img,(x,y),5,(0,0,255),-1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        else:
            cv.circle(img,(x,y),5,(0,0,255),-1)


img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
cv.destroyAllWindows()

# %%
import numpy as np

test = np.array((5,5))

np.floor(np.array(test.shape)/2).astype(int)

# %%
import cv2

vid_read = cv2.VideoCapture('/mnt/Kennedy_SMB/ASAP/iMCI-P60induction/2596506_5678/20231024/openfield/Basler_acA1300-60gm__24108332__20231024_111226468.mp4')


# %%
from os import path
import random
import numpy as np

# %%
vid_dirname, vid_filepath = path.split('/mnt/Kennedy_SMB/ASAP/iMCI-P60induction/2596506_5678/20231024/openfield/Basler_acA1300-60gm__24108332__20231024_111226468.mp4')
vid_basename = path.splitext(vid_filepath)[0]

label_frames = random.choices(range(int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))), k=int(np.min()))

# %%
'height: 982, width: 1517'
{'center': 319, 'east': 350, 'north': 251, 'south': 381, 'west': 336}
{'center': 503, 'east': 548, 'north': 520, 'south': 524, 'west': 445}
{'center': [822, 1265, 1141, 1768],
 'east': [316, 969, 666, 1517],
 'north': [0, 498, 251, 1018],
 'south': [601, 496, 982, 1020],
 'west': [323, 0, 659, 445]}
{'center': array([ 305, 1526,  624, 2029]),
 'east': array([ 284, 2046,  634, 2594]),
 'north': array([  30, 1512,  281, 2032]),
 'south': array([ 637, 1526, 1018, 2050]),
 'west': array([ 298, 1067,  634, 1512])}

# %%
with open('/home/klb807/Documents/AWS_labeling_setup/boundaries.txt','r+') as fid:
    contents = fid.readlines()


# %%
import generate_AWS_multiview_template

keypoints = ["Nose","Throat","Body Center","Right Ear","Left Ear","RIght Hip","Left Hip","Tail Base","Tail Mid","Tail Tip"]

generate_AWS_multiview_template.generate_AWS_template(keypoints,'/home/klb807/Documents/')

# %%
from matplotlib import pyplot as plt
import cv2
import numpy as np

# %%
vid_read = cv2.VideoCapture('C:/Users/17204/Documents/Data/cropped/Basler_acA1300-60gm__24254439__20231026_105544912_cropped.mp4')
ret,frame = vid_read.read()

l1 = 'Label each keypoint:'
l2 = '    - in at least 3 views'
l3 = '    - only once per view'
b1 = 'Refer to instructions'
b2 = 'for view layout'
inst_scale = 0.5
top_size = cv2.getTextSize(l1, cv2.FONT_HERSHEY_SIMPLEX, inst_scale,1)[0]
b1_size = cv2.getTextSize(b1, cv2.FONT_HERSHEY_SIMPLEX, inst_scale,1)[0]
b2_size = cv2.getTextSize(b2, cv2.FONT_HERSHEY_SIMPLEX, inst_scale,1)[0]

# put a header onto the frame
header = np.zeros((5*top_size[1],frame.shape[1],frame.shape[2]))
frame_new = np.concatenate((header,frame), axis=0)
# and a footer
footer = np.zeros((2*b1_size[1], frame_new.shape[1], 3))
frame_new = np.concatenate((frame_new, footer),axis=0).astype(np.uint8)

b1_origin = (int(frame.shape[1]-(b1_size[0]+5)),int(frame.shape[0] - 2*b1_size[1]))
b2_origin = (int(frame.shape[1]-(b2_size[0]+5)),int(frame.shape[0] - 0.5*b2_size[1]))
cv2.putText(frame, l1, (5,int(1.5*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
cv2.putText(frame, l2, (5,int(3*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
cv2.putText(frame, l3, (5,int(4.5*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
cv2.putText(frame, b1, b1_origin, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
cv2.putText(frame, b2, b2_origin, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
# cv2.putText(frame_new, l1, (5,int(1.5*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
# cv2.putText(frame_new, l2, (5,int(3*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
# cv2.putText(frame_new, l3, (5,int(4.5*top_size[1])), cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
# cv2.putText(frame_new, bottom_text, bottom_origin, cv2.FONT_HERSHEY_SIMPLEX, inst_scale, (255,255,255))
                

fig,ax = plt.subplots(ncols=2)
ax[0].imshow(frame)
# ax[1].imshow(frame_new)

cv2.imwrite('C:/Users/17204/Documents/Data/Cropped/PutText_Concat.png',frame)

# %%
import cv2 as cv

# %%
img = cv.imread('/home/klb807/git/Work_Tools/Reminder.drawio.png')

while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27: # press esc to esc
        break
cv.destroyAllWindows()

# %%
img_thresh = ((img > 125)*255).astype(np.uint8)
np.zeros((frame.shape[0] + top_size[1]*5,frame.shape[1],frame.shape[2])).shape

# %%
while(1):
    cv.imshow('image',img_thresh)
    k = cv.waitKey(1) & 0xFF
    if k == 27: # press esc to esc
        break
cv.destroyAllWindows()

# %% [markdown]
# 


