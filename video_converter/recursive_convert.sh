#/bin/bash

for f in *.tiff; do echo "file '$f'\rduration 0.0166666" >> convert_list.txt; done
ffmpeg -f concat -i convert_list.txt -framerate 60 -s 1280x1024 1024.mp4

rm convert_list.txt
