"""
the experiment for examining the effect of ROI selection under different subject's motion types.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(metric, list_motion, list_algorithm):
    """main function to evaluate the effect of ROI selection under different subject's motion types.
    Parameters
    ----------
    metric: evaluation metric.   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_motion: list of subject's motion types.   # ['resting', 'gym', 'rotation', 'talk'].
    list_algorithm: list of selecte rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].
    
    Returns
    -------

    """
    # get project directory.
    dir_crt = os.getcwd()
    data_rank = np.zeros(28)
    # loop over all motion types.
    for i_motion in range(len(list_motion)):
        motion = list_motion[i_motion]
        # loop over all selected algorithms and then compute the average.
        for i_algorithm in range(len(list_algorithm)):
            algorithm = list_algorithm[i_algorithm]
            # metric dataframe.
            # LGI-PPGI dataset.
            dir_LGI_PPGI = os.path.join(dir_crt, 'result', 'LGI-PPGI', 'evaluation_'+algorithm+'.csv')
            df_LGI_PPGI = pd.read_csv(dir_LGI_PPGI, index_col=0)
            df_LGI_PPGI = df_LGI_PPGI.loc[df_LGI_PPGI['motion'].values==motion, :]
            df_LGI_PPGI = df_LGI_PPGI.reset_index()
            # list of ROI names.
            list_roi_name = (np.unique(df_LGI_PPGI['ROI'].values)).tolist()
            # collect performance evaluation results.
            if i_algorithm == 0:
                data_roi = []
            for i in range(len(list_roi_name)):
                # LGI-PPGI dataset.
                metric_tmp = np.median(df_LGI_PPGI.loc[df_LGI_PPGI['ROI'].values==list_roi_name[i], metric].dropna().values)
                if i_algorithm == 0:
                    data_roi.append(np.array(metric_tmp))
                else:
                    data_roi[i] = data_roi[i] + metric_tmp
        # transform into ranking.
        data_rank = data_rank + (pd.Series(data_roi)).rank()
    data_rank = data_rank/len(list_motion)
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
    dir_save = os.path.join(dir_crt, 'plot', 'motion', metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # print ranking.
    if metric in ['PCC', 'CCC']:
        list_roi_name.reverse()
    print(list_roi_name)


if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_metric = ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW']   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_motion = ['gym', 'rotation', 'talk']   # ['resting', 'gym', 'rotation', 'talk'].
    # loop over all performance evaluation metrics.
    for metric in list_metric:
        print(metric)
        main(metric=metric, list_motion=list_motion, list_algorithm=list_algorithm)