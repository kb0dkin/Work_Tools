
import os
from shutil import copy2
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np



file_walk = os.walk("/mnt/Kennedy_SMB/ASAP/iMCI-P30induction")
for dirpath,dirname,filenames in file_walk:

    tiff_files = [file for file in filenames if '.tiff' in file]

    if tiff_files:
        old_dir = os.getcwd()
        # if there are any tiffs in the directory
        os.chdir(dirpath) # move to the directory -- this makes things easier


        dirsplit = dirpath.split(os.sep)
        mouse = dirsplit[-3]
        rec_date = dirsplit[-2]
        task = dirsplit[-1]

        vid_name = '_'.join([mouse,rec_date,task])+'.mp4'
        
        tiff_suffix = [int(tiff[:-5].split('_')[-1]) for tiff in tiff_files]
        sort_ind = np.argsort(np.array(tiff_suffix))
        sort_tiff = [tiff_files[ind] for ind in sort_ind]
        sort_suffix = np.array([tiff_suffix[ind] for ind in sort_ind])

        with open('framelist.txt', 'w') as fid:
            fid.write("".join([f"file '{file}'\nduration 0.02\n" for file in sort_tiff]))

        # # create the video
        os.popen(f'''ffmpeg -f concat -safe 0 -r 50 -i {'framelist.txt'} -r 50\
                {vid_name}
        ''').read()

        # stick into a zip archive
        zip_name = 'images.zip'
        with ZipFile(zip_name,'w', compression=ZIP_DEFLATED, compresslevel=9) as zf:
            for file in tiff_files:
                zf.write(file)

        os.remove('framelist.txt')

        # copy everything to the server
        # dest_name = 'Y:\\ASAP\\'+ mouse + '\\' + rec_date + '\\' + task
        # if not os.path.exists(dest_name):
            # os.mkdir(dest_name)
        # copy2(zip_name, dest_name)
        # copy2(vid_name,dest_name)


        print(vid_name)
        os.chdir(old_dir)


