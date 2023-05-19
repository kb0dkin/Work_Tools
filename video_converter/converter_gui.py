# converter_gui.py
#
# Converts .tiff files to either mp4 or avi using ffmpeg
# on the backend. Basic GUI allows the user to select the
# import and save folder.
#
# KLB April 2023


# import things
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import os
import re
import glob
import time


class converter_gui(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent)        # select directory for individual tiffs

        # input directory selection
        self.tiff_dn = tk.StringVar() # name of the input directory
        self.tiff_button = ttk.Button(self, text='Select Input Directory', command=self.get_dir)
        self.tiff_button.grid(row=0, column=0, padx=5, pady=5)        # # select save filename
        
        # save filename info -- not for selection
        self.vid_name_frame = tk.LabelFrame(self, text='Output video names')
        self.save_fn = tk.StringVar()
        self.save_entry = ttk.Label(self.vid_name_frame, textvariable=self.save_fn)
        self.save_entry.grid(row=0, column=0, padx=5, pady=5)        # convert button
        self.vid_name_frame.grid(row=1,column=0, padx=5, pady=5)
        self.vids = {}
        
        # conversion button
        self.convert_button = ttk.Button(self, text='Convert Files', command=self.convert_files)
        self.convert_button.grid(row=2, column=0)        # get it to run
        
        self.grid(row=0, column=0)    # define convert command


    # convert to mp4/avi
    def convert_files(self):
        t = time.time()


        # for each video
        for vid_name,vid_path in self.vids.items():
            # get all frames that match, sort by creation time
            files = [file for file in glob.glob(vid_path + '*.tiff')]
            files.sort()

            # create the frame list file
            with open(vid_path+'_framelist.txt', 'w') as fid:
                for file in files:
                    fid.write(f"file '{file}'\nduration 0.02\n")

            # create the movie using the list file
            # os.popen(f'''ffmpeg -framerate 60 -f image2 -i {base_fn}_%10d.tiff -s 1280x1024 {base_fn}_1024.mp4\
            #         -s 1280x1024 {base_fn}_1024.mp4\
            #         -s 640x512 {base_fn}_512.mp4\
            #         -s 1280x1024 {base_fn}_1024.avi\
            #         -s 640x512 {base_fn}_512.avi''')
            os.popen(f'''ffmpeg -f concat -safe 0 -i {vid_path+'_framelist.txt'} -framerate 50\
                    -s 1280x1024 {vid_path}_1024.mp4
            ''').read()

            print(f"created video {vid_path}_1024.mp4")
            os.remove(vid_path+'_framelist.txt')

        print(f"Elapsed time: {time.time()-t}")
        
        self.destroy()
        root.destroy()



    # get the directory -- due to how the callbacks work
    def get_dir(self):
        self.tiff_dn.set(fd.askdirectory())
        
        read_dir = self.tiff_dn.get()
        # check if the input directory exists
        if not os.path.exists(read_dir):
            print('The input directory does not exist!')        # parse save_fn as needed

        # get the list of files
        vids = {}
        for root,_,files in os.walk(read_dir): # recursively run through subdirs
            temp_vids = list(set([re.findall('(.+)_\d+.tiff$', file)[0] for file in files if 'tiff' in file]))
            for vid in temp_vids:
                vids[vid] = os.path.join(root,vid)

        # get the base filename -- without the image counts
        self.vids = vids

        # update the GUI to list the videos 
        self.save_fn.set('\n'.join(list(vids.keys()))) # convert the list into a multi-line string


if __name__ == "__main__":
    # root window
    root = tk.Tk()
    frm = converter_gui(root)
    root_frame = tk.Frame(root)    # build it
    root.mainloop()