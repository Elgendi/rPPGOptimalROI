"""
the experiment for examining the effect of ROI selection under different light intensity.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main_illumination_rank(metric, list_roi_name, list_algorithm):
    """main function to evaluate the effect of ROI selection under different illumiance levels.
    Parameters
    ----------
    metric: evaluation metric.   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_roi_name: list of selected ROIs.   # 28 facial ROIs.
    list_algorithm: list of selected rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].
    
    Returns
    -------

    """
    # get current directory.
    dir_crt = os.getcwd()
    # lux level. 
    list_lux = [10.0, 15.8 ,25.1, 39.8, 63.1, 100.0]
    # data array initialization.
    data_roi = np.zeros(shape=[len(list_lux), len(list_roi_name)])
    # loop over all selected algorithms and then compute the average.
    for algorithm in list_algorithm:
        # metric dataframe.
        dir_BUAA_MIHR = os.path.join(dir_crt, 'result', 'BUAA-MIHR', 'evaluation_'+algorithm+'.csv')
        df_BUAA_MIHR = pd.read_csv(dir_BUAA_MIHR, index_col=0)
        df_BUAA_MIHR = df_BUAA_MIHR.reset_index()
        # loop over all selected facial ROIs.
        for i_roi in range(len(list_roi_name)):
            ROI = list_roi_name[i_roi]
            # loop over all illuminance levels.
            for i_lux in range(len(list_lux)):
                data_tmp = df_BUAA_MIHR.loc[(df_BUAA_MIHR['ROI'].values == ROI)&(df_BUAA_MIHR['lux'].values == list_lux[i_lux]), metric].values
                data_tmp = data_tmp[~np.isnan(data_tmp)]
                data_roi[i_lux, i_roi] = data_roi[i_lux, i_roi] + np.mean(data_tmp)
    # take average over all algorithms.
    data_roi = data_roi/len(list_algorithm)
    # transform into ranking.
    data_rank = np.zeros(len(list_roi_name))
    for i in range(len(list_lux)):
        data_rank = data_rank + (pd.Series(data_roi[i, :])).rank()
    data_rank = data_rank/len(list_lux)
    # sorting.
    idx_sort = np.argsort(data_rank)
    data_rank = (np.array(data_rank)[idx_sort])
    list_roi_name = (np.array(list_roi_name)[idx_sort]).tolist()
    # visualization and save data.
    plt.cla()
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.bar(range(len(list_roi_name)), data_rank, tick_label=list(range(1, len(list_roi_name)+1)), color='black')
    dir_save = os.path.join(dir_crt, 'plot', 'illumination', metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # print ranking.
    if metric in ['PCC', 'CCC']:
        list_roi_name.reverse()
    print(list_roi_name)


if __name__ == "__main__":
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_metric = ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW']   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_roi_name = ['lower medial forehead', 
                     'left lower lateral forehead', 
                     'right lower lateral forehead', 
                     'glabella', 
                     'upper nasal dorsum', 
                     'lower nasal dorsum', 
                     'soft triangle', 
                     'left ala', 
                     'right ala', 
                     'nasal tip', 
                     'left lower nasal sidewall', 
                     'right lower nasal sidewall', 
                     'left mid nasal sidewall', 
                     'right mid nasal sidewall', 
                     'philtrum', 
                     'left upper lip', 
                     'right upper lip', 
                     'left nasolabial fold', 
                     'right nasolabial fold', 
                     'left temporal', 
                     'right temporal', 
                     'left malar', 
                     'right malar', 
                     'left lower cheek', 
                     'right lower cheek', 
                     'chin', 
                     'left marionette fold', 
                     'right marionette fold']   # 28 facial ROIs.
    # loop over all selected evaluation metrics.
    for metric in list_metric:
        print([metric])
        main_illumination_rank(metric=metric, list_roi_name=list_roi_name, list_algorithm=list_algorithm)