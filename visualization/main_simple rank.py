"""
the experiment for examining the effect of ROI selection under different subject's motion types.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(metric, list_algorithm):
    """main function to evaluate the effect of ROI selection under different subject's motion types.
    Parameters
    ----------
    metric: evaluation metric.   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_algorithm: list of selecte rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].
    
    Returns
    -------

    """
    # get project directory.
    dir_crt = os.getcwd()
    # fringe list.
    list_UBFC_RPPG_fringe = [5, 11, 15, 30, 32, 38]
    list_UBFC_PHYS_fringe = [27, 30, 47]
    # beard list.
    list_UBFC_RPPG_beard = [3, 12, 13, 14, 23, 24, 30, 31, 32, 39, 44]
    list_UBFC_PHYS_beard = [2, 19, 38, 48, 49]
    # loop over all selected algorithms and then compute the average.
    for i_algorithm in range(len(list_algorithm)):
        algorithm = list_algorithm[i_algorithm]
        # metric dataframe.
        # UBFC-rPPG dataset.
        dir_UBFC_RPPG = os.path.join(dir_crt, 'result', 'UBFC-rPPG', 'evaluation_'+algorithm+'.csv')
        df_UBFC_RPPG = pd.read_csv(dir_UBFC_RPPG, index_col=0)
        # UBFC-Phys dataset.
        dir_UBFC_PHYS = os.path.join(dir_crt, 'result', 'UBFC-Phys', 'evaluation_'+algorithm+'.csv')
        df_UBFC_PHYS = pd.read_csv(dir_UBFC_PHYS, index_col=0)

        # collect data of subjects with beard.
        # UBFC-rPPG dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_RPPG)):
            if df_UBFC_RPPG.loc[i, 'attendant'] in list_UBFC_RPPG_beard:
                list_idx_tmp.append(False)
            else:
                list_idx_tmp.append(True)
        df_UBFC_RPPG_beard_pure = df_UBFC_RPPG.loc[list_idx_tmp, :]
        df_UBFC_RPPG_beard_pure = df_UBFC_RPPG_beard_pure.reset_index()
        # UBFC-Phys dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_PHYS)):
            if df_UBFC_PHYS.loc[i, 'attendant'] in list_UBFC_PHYS_beard:
                list_idx_tmp.append(False)
            else:
                list_idx_tmp.append(True)
        df_UBFC_PHYS_beard_pure = df_UBFC_PHYS.loc[list_idx_tmp, :]
        df_UBFC_PHYS_beard_pure = df_UBFC_PHYS_beard_pure.reset_index()

        # collect data of subjects with fringe.
        # UBFC-rPPG dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_RPPG)):
            if df_UBFC_RPPG.loc[i, 'attendant'] in list_UBFC_RPPG_fringe:
                list_idx_tmp.append(False)
            else:
                list_idx_tmp.append(True)
        df_UBFC_RPPG_fringe_pure = df_UBFC_RPPG.loc[list_idx_tmp, :]
        df_UBFC_RPPG_fringe_pure = df_UBFC_RPPG_fringe_pure.reset_index()
        # UBFC-Phys dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_PHYS)):
            if df_UBFC_PHYS.loc[i, 'attendant'] in list_UBFC_PHYS_fringe:
                list_idx_tmp.append(False)
            else:
                list_idx_tmp.append(True)
        df_UBFC_PHYS_fringe_pure = df_UBFC_PHYS.loc[list_idx_tmp, :]
        df_UBFC_PHYS_fringe_pure = df_UBFC_PHYS_fringe_pure.reset_index()

        # list of ROI names.
        list_roi_name = (np.unique(df_UBFC_RPPG['ROI'].values)).tolist()
        # collect performance evaluation results.
        if i_algorithm == 0:
            data_roi = []
        for i_roi in range(len(list_roi_name)):
            roi_name = list_roi_name[i_roi]
            if roi_name in ['lower medial forehead', 'left lower lateral forehead', 'right lower lateral forehead']:
                metric_tmp = np.concatenate((df_UBFC_RPPG_fringe_pure.loc[df_UBFC_RPPG_fringe_pure['ROI'].values==list_roi_name[i_roi], metric].values, 
                                             df_UBFC_PHYS_fringe_pure.loc[df_UBFC_PHYS_fringe_pure['ROI'].values==list_roi_name[i_roi], metric].values), 
                                             axis=0)
            elif roi_name in ['philtrum', 'left upper lip', 'right upper lip', 'left nasolabial fold', 
                              'right nasolabial fold', 'chin', 'left marionette fold', 'right marionette fold']:
                metric_tmp = np.concatenate((df_UBFC_RPPG_beard_pure.loc[df_UBFC_RPPG_beard_pure['ROI'].values==list_roi_name[i_roi], metric].values, 
                                             df_UBFC_PHYS_beard_pure.loc[df_UBFC_PHYS_beard_pure['ROI'].values==list_roi_name[i_roi], metric].values), 
                                             axis=0)
            else:
                metric_tmp = np.concatenate((df_UBFC_RPPG.loc[df_UBFC_RPPG['ROI'].values==list_roi_name[i_roi], metric].values, 
                                             df_UBFC_PHYS.loc[df_UBFC_PHYS['ROI'].values==list_roi_name[i_roi], metric].values), 
                                             axis=0)
            
            if i_algorithm == 0:
                data_roi.append(metric_tmp)
            else:
                data_roi[i_roi] = data_roi[i_roi] + metric_tmp
    # average over all included algorithms.
    data_roi = (np.array(data_roi, dtype=object)/len(list_algorithm)).tolist()
    # sorting.
    list_median = []
    for data in data_roi:
        list_median.append(np.median(data))
    if metric in ['MAE', 'RMSE', 'DTW']:
        idx_sort = np.argsort(list_median)
    else:
        idx_sort = np.argsort(list_median)[::-1]
    data_roi = (np.array(data_roi, dtype=object)[idx_sort]).tolist()
    list_roi_name = (np.array(list_roi_name, dtype=object)[idx_sort]).tolist()
    # visualization and save data.
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.cla()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.boxplot(x=data_roi, zorder=True, labels=list(range(1, len(list_roi_name)+1)))
    dir_save = os.path.join(dir_crt, 'plot', 'simple', metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # print ranking.
    print(list_roi_name)


if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_metric = ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW']   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    # loop over all performance evaluation metrics.
    for metric in list_metric:
        print(metric)
        main(metric=metric, list_algorithm=list_algorithm)