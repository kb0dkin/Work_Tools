
import os
from shutil import copy2
from zipfile import ZipFile, ZIP_DEFLATED



file_walk = os.walk("E:")

for dirpath,dirname,filenames in file_walk:

    tiff_files = [file for file in filenames if '.tiff' in file]

    if tiff_files:
        old_dir = os.getcwd()
        # if there are any tiffs in the directory
        os.chdir(dirpath) # move to the directory -- this makes things easier


        dirsplit = dirpath.split('\\')
        mouse = dirsplit[-3]
        rec_date = dirsplit[-2]
        task = dirsplit[-1]

        vid_name = '_'.join([mouse,rec_date,task])+'.mp4'

        tiff_files.sort()
        # print("".join([f"file '{file}'\nduration 0.02\n" for file in filenames if ".png" in file]))
        with open('framelist.txt', 'w') as fid:
            fid.write("".join([f"file '{file}'\nduration 0.02\n" for file in filenames if ".tiff" in file]))

        # # create the video
        os.popen(f'''ffmpeg -f concat -safe 0 -i {'framelist.txt'} -framerate 50\
                -s 1280x1024 {vid_name}
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