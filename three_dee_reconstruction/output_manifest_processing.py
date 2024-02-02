import argparse
import os
import json



# pull in a multi-view output manifest from AWS, then output the clipped views and 
# associated labels into something that MARS can take in

# takes a manifest and image directory, creates a sub direcory, splits the images
# and outputs the MARS-friendly keypoints
def output_manifest_processing(manifest_path:str, image_directory:str):

    # does the manifest exist?
    if not os.path.exists(manifest_path):
        print(f'Manifest file {manifest_path} does not exist')

    # does the manifest exist?
    if not os.path.exists(image_directory):
        print(f'Directory {image_directory} does not exist')


    # Parse the manifest file
    with open(manifest_path, 'r') as fid:
        anns = []
        for line in fid.readlines():
            # anns.append(json.loads(line))
            print(json.loads(line)['source-ref'])

    # print(anns)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_path')
    parser.add_argument('image_directory', default='.')

    args = parser.parse_args()

    output_manifest_processing(args.manifest_path, args.image_directory)
