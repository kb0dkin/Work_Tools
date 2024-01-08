
import os
from shutil import copy2
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import argparse


def main(walk_dir='/media/fsmresfiles/ASAP/iMCI-P60induction/', log_file = '/home/kevin/Documents/convert_log.txt', max_directories = -1):

    print(f'Looking through {walk_dir} for images and videos...')
    file_walk = os.walk(walk_dir)

    # create a list of the directories -- just so we can have a status bar later
    convert_list = []
    done_list = []
    for dirpath,_,filenames in file_walk:
        tiff_files = [file for file in filenames if '.tiff' in file]
        mp4_files = [file for file in filenames if '.mp4' in file]

        # skip this directory if it doesn't have images
        if not tiff_files:
            continue

        # append to the done_list if it already has videos
        if mp4_files:
            done_list.append(dirpath)
            continue

        convert_list.append(dirpath)


    # give a heads up about the number of directories to convert etc
    print(f'Found {len(convert_list)} directories with tiffs that need to be converted')
    print(f'Found {len(done_list)} directories already completed')
    print(f'Planning to convert {max_directories if max_directories > -1 else len(convert_list)}')

    convert_confirm = input('Do you want to continue ([y]/n)')

    if convert_confirm.lower() in ['n','no']:
        print('Stopping conversion')
        return
    
    for i_conv, dirpath in enumerate(convert_list):
        if i_conv == max_directories:
            break

        print(f'{i_conv+1} of {len(convert_list)}: {dirpath}')

        old_dir = os.getcwd()
        os.chdir(dirpath) # move to the directory -- this makes things easier

        dirsplit = dirpath.split(os.sep)
        mouse = dirsplit[-3]
        rec_date = dirsplit[-2]
        task = dirsplit[-1]

        # future video name
        vid_name = '_'.join([mouse,rec_date,task])+'.mp4'

        # tiff files in the directory
        tiff_files = [file for file in os.listdir('.') if '.tiff' in file]

        # sort the filenames properly
        tiff_split = np.array([tiff.split('_')[-1].split('.')[0] for tiff in tiff_files]).astype(int)
        tiff_sort = np.array(tiff_files)[np.argsort(tiff_split)].tolist()


        with open('framelist.txt', 'w') as fid:
            fid.write("".join([f"file '{file}'\nduration 0.02\n" for file in tiff_sort]))

        # # create the video
        os.popen(f'''ffmpeg -f concat -safe 0 -r 50 -i {'framelist.txt'} -r 50\
                {vid_name} -hide_banner -loglevel warning
        ''').read()


        os.remove('framelist.txt')


        os.chdir(old_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='recursively walks a directory and converts subdirectories of tiff files into videos at 50 hz')
    parser.add_argument('-b','--base_dir',default='/media/fsmresfiles/ASAP/iMCI-P60induction/', help='base directory to walk')
    parser.add_argument('-l','--log',default='/home/kevin/Documents/convert_log.txt',help='file to store the outputs of the ffmpeg calls')
    parser.add_argument('-m','--max_vids',type=int, default=-1)

    args = parser.parse_args()
    main(walk_dir=args.base_dir, log_file=args.log, max_directories=args.max_vids)
