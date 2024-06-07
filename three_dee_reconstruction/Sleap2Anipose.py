#! /usr/bin/env python

# Sleap2Anipose
'''
Converts a .slp labels file into a DeepLabCut hdf5 (from Pandas)

The DLC pandas dataframes are multi-index;
    1. Scorer: Labeling source (individual or model)
    2. BodyParts
    3. coordinates (x, y and confidence)

'''

import sleap
import pandas as pd
import os
import argparse


def Sleap2Anipose(slp_file:str, output_path: None, csv:bool = False):
    '''
    Converts a .slp file to a Anipose/DLC hdf5

    Arguments:
        - slp_file      Path to .slp annotation file
        - output_path   (optional) Path to save hdf5 file. will save to <slp_file>.h5 otherwise
    '''

    # does the slp file exist?
    if not os.path.exists(slp_file):
        print('{slp_file} not found')
        return -1
    
    # create h5 path if doesn't exist
    if output_path is None:
        output_path = os.path.splitext(slp_file)[0] + '.h5'

    # load the data
    slp_data = sleap.load_file(slp_file)

    # pull out the label points
    nodes = [node.name for node in slp_data.skeleton.nodes]

    # create the MultiIndex
    mInd = pd.MultiIndex.from_product([['sleap'],nodes,['x','y','likelihood']], names=['scorer','bodyparts','coords'])

    # iterate through the instances
    slp_df = pd.DataFrame(columns=mInd, index=range(slp_data[-1].frame_idx + 1))
    for frame in slp_data:
        # create an empty entry to append...
        entry = {}

        # use the user label if available, otherwise use the predicted
        if frame.has_user_instances:
            temp_instance = frame.user_instances[0]
            inst_type = 'user'
        elif frame.has_predicted_instances:
            temp_instance = frame.predicted_instances[0]
            inst_type = 'pred'
        else: # insert a blank row to keep the labels aligned with the video frames
            continue


        # loop through the available nodes
        for node in [node.name for node in temp_instance.nodes]:
            if temp_instance[node]['visible']:
                entry[('sleap',node,'x')] = temp_instance[node]['x']
                entry[('sleap',node,'y')] = temp_instance[node]['y']
                entry[('sleap',node,'likelihood')] = temp_instance[node]['score'] if inst_type == 'pred' else 1.0
        
        slp_df.loc[frame.frame_idx,:] = entry

    

    
    # need to replace all of the NaNs with 0s
    slp_df.fillna(0.0, inplace=True)

    # hacky code -- delete the tail middle and tail base columns
    # slp_df.drop(columns='Tail Middle', level=1, inplace=True)
    # slp_df.drop(columns='Tail Tip', level=1, inplace=True)
            

    # write it to hdf5
    slp_df.to_hdf(output_path, key='df_with_missing')

    if csv:
        csv_path = os.path.splitext(slp_file)[0] + '.csv'
        slp_df.to_csv(csv_path)




# call it from the shell
if __name__ == '__main__':
    docstring = '''
    Converts a .slp file to a Anipose/DLC hdf5
    '''

    parser = argparse.ArgumentParser(description=docstring)

    parser.add_argument('slp_file',help='path to .slp annotation file')
    parser.add_argument('-o', dest='output_path', default=None, help='hdf5 file path')
    parser.add_argument('-c', action='store_true', dest='csv', help='store a .csv copy of dataframe')

    args = parser.parse_args()

    Sleap2Anipose(args.slp_file, args.output_path, args.csv)
            
