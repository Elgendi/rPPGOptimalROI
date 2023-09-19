"""
The experiment for examining the effect of ROI selection under different subject's motion types.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(metric, motion, list_algorithm):
    """main function to evaluate the effect of ROI selection under different subject's motion types.
    Parameters
    ----------
    metric: selecte evaluation metric.   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    motion: subject's motion type.   # ['resting', 'gym', 'rotation', 'talk'].
    list_algorithm: list of selecte rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].
    
    Returns
    -------

    """
    # get project directory.
    dir_crt = os.getcwd()
    # loop over all selected algorithms and then compute the average.
    for i_algorithm in range(len(list_algorithm)):
        algorithm = list_algorithm[i_algorithm]
        # metric dataframe.
        # LGI-PPGI dataset.
        dir_LGI_PPGI = os.path.join(dir_crt, 'result', 'LGI-PPGI', 'evaluation_'+algorithm+'.csv')
        df_LGI_PPGI = pd.read_csv(dir_LGI_PPGI, index_col=0)
        df_LGI_PPGI = df_LGI_PPGI.loc[df_LGI_PPGI['motion'].values==motion, :]
        df_LGI_PPGI = df_LGI_PPGI.reset_index()
        # UBFC-rPPG dataset.
        dir_UBFC_RPPG = os.path.join(dir_crt, 'result', 'UBFC-rPPG', 'evaluation_'+algorithm+'.csv')
        df_UBFC_RPPG = pd.read_csv(dir_UBFC_RPPG, index_col=0)
        # UBFC-Phys dataset.
        dir_UBFC_PHYS = os.path.join(dir_crt, 'result', 'UBFC-Phys', 'evaluation_'+algorithm+'.csv')
        df_UBFC_PHYS = pd.read_csv(dir_UBFC_PHYS, index_col=0)
        # list of ROI names.
        list_roi_name = (np.unique(df_LGI_PPGI['ROI'].values)).tolist()
        # collect performance evaluation results.
        if i_algorithm == 0:
            data_result = []
        for i in range(len(list_roi_name)):
            # LGI-PPGI dataset.
            metric_tmp = df_LGI_PPGI.loc[df_LGI_PPGI['ROI'].values==list_roi_name[i], metric].values
            if motion == 'resting':
                metric_tmp = np.concatenate((metric_tmp, 
                                            df_UBFC_RPPG.loc[df_UBFC_RPPG['ROI'].values==list_roi_name[i], metric].values, 
                                            df_UBFC_PHYS.loc[df_UBFC_PHYS['ROI'].values==list_roi_name[i], metric].values), 
                                            axis=0)
            if i_algorithm == 0:
                data_result.append(np.array(metric_tmp))
            else:
                data_result[i] = data_result[i] + np.array(metric_tmp)
    # sorting.
    idx_sort = np.argsort(np.median(np.array(data_result), axis=1))
    data_result = (np.array(data_result)[idx_sort]).tolist()
    list_roi_name = (np.array(list_roi_name)[idx_sort]).tolist()
    # visualization and save data.
    plt.cla()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.boxplot(x=data_result, zorder=True, labels=list(range(1, len(list_roi_name)+1)))
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    dir_save = os.path.join(dir_crt, 'plot', 'motion', motion+'_'+metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # write description file.
    with open(os.path.join(dir_crt, 'plot', 'motion', 'description.txt'), 'a') as f:
        f.writelines([motion, '\n', metric, '\n', str(list_roi_name), '\n'])


if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_metric = ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW']
    list_motion = ['resting', 'gym', 'rotation', 'talk']
    # create the description file.
    with open(os.path.join(os.getcwd(), 'plot', 'motion', 'description.txt'), 'w') as f:
        pass
    # loop over all performance evaluation metrics.
    for metric in list_metric:
        # loop over all motion types.
        for motion in list_motion:
            print([metric, motion])
            main(metric=metric, motion=motion, list_algorithm=list_algorithm)