"""
Transform RGB signals to BVP and HR signals.
"""

# Author: Shuo Li
# Date: 2023/07/18

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pre_analysis
import pyVHR
from tqdm import tqdm


def main_rgb2hr(name_dataset, algorithm):
    """Main function for transforming RGB traces to HR-related signals.
    Parameters
    ----------
    name_dataset: name of the selected dataset.
                  [UBFC-rPPG, UBFC-Phys, LGI-PPGI, BH-rPPG, ECG-fitness].
    algorithm: selected rPPG algorithm. ['CHROM', 'GREEN', 'ICA', 'LGI', 'OMIT', 'PBV', 'PCA', 'POS'].
    
    Returns
    -------

    """
    # get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # parameter class initialization.
    Params = util_pre_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)

    # RGB signal -> bvp signal.
    if name_dataset == 'UBFC-rPPG':
        # sequnce num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
        # loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # parse the RGB signal from the RGB dataframe. size = [num_frames, num_ROI, rgb_channels(3)].
            dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'.csv')
            df_rgb = pd.read_csv(dir_sig_rgb)
            # RGB signal initialization.
            sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
            # loop over all ROIs.
            for i_roi in range(len(Params.list_roi_name)):
                sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # red channel.
                sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # green channel.
                sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # blue channel.
            # RGB video information.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            # get video fps.
            capture = cv2.VideoCapture(dir_vid)
            Params.fps = capture.get(cv2.CAP_PROP_FPS)
            # RGB signal -> bvp signal & bpm signal.
            sig_bvp, sig_bpm = util_pre_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
            # create the dataframe to save the HR-related data (bvp signal & bpm signal).
            df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
            df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
            # loop over all ROIs.
            for i_roi in range(len(Params.list_roi_name)):
                df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
            # data saving.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant)+'_'+algorithm+'.csv')
            df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == 'UBFC-Phys':
        # list of attendants.
        list_attendant = list(range(1, 57))
        # condition types.
        list_condition = [1]   # [1, 2, 3]
        for num_attendant in tqdm(list_attendant):
            for condition in list_condition:
                # parse the RGB signal from the RGB dataframe. size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant) + '_' + str(condition) + '.csv')
                df_rgb = pd.read_csv(dir_sig_rgb)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(condition)+'.avi')
                # get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_pre_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant) + '_' + \
                                             str(condition) + '_' + algorithm + '.csv')
                df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == 'LGI-PPGI':
        # name of attendants.
        list_attendant = ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun']
        # motion types.
        list_motion = ['gym', 'resting', 'talk', 'rotation']
        for attendant in tqdm(list_attendant):
            for motion in list_motion:
                # parse the RGB signal from the RGB dataframe. size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', attendant+'_'+motion+'.csv')
                df_rgb = pd.read_csv(dir_sig_rgb)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']   # red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']   # green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']   # blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, attendant, attendant+'_'+motion, 'cv_camera_sensor_stream_handler.avi')
                # get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_pre_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', attendant+'_'+motion+'_'+algorithm+'.csv')
                df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == 'BUAA-MIHR':
        # sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # lux levels.
        list_lux = ['lux 10.0', 'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0']
        # attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # loop over all illumination levels.
            for lux in list_lux:
                # parse the RGB signal from the RGB dataframe. size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant).zfill(2)+'_'+lux.replace(' ', '')+'.csv')
                df_rgb = pd.read_csv(dir_sig_rgb, index_col=None)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, 'Sub '+str(num_attendant).zfill(2), lux, \
                                       lux.replace(' ', '') + '_' + list_name[num_attendant-1]+'.avi')
                # get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_pre_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant).zfill(2) + \
                                             '_' + str(lux).replace(' ', '') + '_' + algorithm+'.csv')
                df_hr.to_csv(dir_save_data, index=False)


if __name__ == "__main__":
    # available datasets.
    list_dataset = ['UBFC-Phys']
    # selected rPPG algorithms.
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']
    for name_dataset in list_dataset:
        for algorithm in list_algorithm:
            print([name_dataset, algorithm])
            main_rgb2hr(name_dataset=name_dataset, algorithm=algorithm)