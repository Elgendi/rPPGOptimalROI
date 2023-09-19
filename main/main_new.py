"""
A demo of the automatic ROI selection.
"""

# Author: Shuo Li
# Date: 2023/05/05

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pre_analysis
import pyVHR
import pyVHR.BVP.methods as methods
from pyVHR.BVP.BVP import RGB_sig_to_BVP
from pyVHR.extraction.utils import MagicLandmarks
from pyVHR.BVP.filters import apply_filter, BPfilter
from pyVHR.BPM.BPM import BVP_to_BPM, BPM_median
from pyVHR.extraction.sig_processing import SignalProcessing


def main():
    """Main function for face detection of videos.
    Parameters
    ----------
    
    Returns
    -------

    """
    # get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # parameter class initialization.
    Params = util_pre_analysis.Params(dir_option=dir_option, name_dataset='UBFC-rPPG')
    # groundtruth class initialization.
    GT = util_pre_analysis.GroundTruth(Params.dir_dataset, 'UBFC-rPPG')
    gtTime, gtTrace, gtHR = GT.get_GT(['realistic', 1], slice=[0, 1])
    # video -> RGB signal.
    file_vid = 'E:/ECG fitness/09/01/C920-1.avi'
    Params.fps = pyVHR.extraction.utils.get_fps(file_vid)
    sig_processing = SignalProcessing()
    ldmks_list = [MagicLandmarks.cheek_left_top[16], MagicLandmarks.cheek_right_top[14], MagicLandmarks.forehead_center[1]]
    sig_processing.set_landmarks(ldmks_list)
    sig_processing.set_square_patches_side (28.0)
    sig_rgb = sig_processing.extract_patches(file_vid , 'squares', 'mean')
    # RGB signal -> windowed signal.
    sig_win, timeES = util_pre_analysis.sig_to_windowed(sig_rgb=sig_rgb, Params=Params)
    # windowed signal -> bvp signal.
    sig_bvp = RGB_sig_to_BVP(windowed_sig=sig_win, fps=Params.fps, device_type='cuda', method=methods.cupy_POS, params={'fps':Params.fps})
    # signal filtering.
    sig_bvp_filtered = apply_filter(sig_bvp, BPfilter, params={'order':6, 'minHz':0.65, 'maxHz':4.0, 'fps':Params.fps})
    # bvp signal -> bpm signal.
    multi_sig_bpm = BVP_to_BPM(bvps=sig_bvp_filtered, fps=Params.fps, minHz=0.65, maxHz=4.0)
    sig_bpm, MAD = BPM_median(multi_sig_bpm)
    # visualization.
    sig_bpm = np.array(sig_bpm)
    #sig_bpm_0 = sig_bpm_new[:, 0]
    #sig_bpm_1 = sig_bpm_new[:, 1]
    #sig_bpm_2 = sig_bpm_new[:, 2]
    # interpolation.
    #sig_bpm_0_interp = np.interp(x=gtTime, xp=timeES, fp=sig_bpm_0)
    #sig_bpm_1_interp = np.interp(x=gtTime, xp=timeES, fp=sig_bpm_1)
    sig_bpm = np.interp(x=gtTime, xp=timeES, fp=sig_bpm)
    corr_0 = np.corrcoef(gtHR, sig_bpm)
    plt.plot(gtHR, label='HR')
    plt.plot(sig_bpm, label='0')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()