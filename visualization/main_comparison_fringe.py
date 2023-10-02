"""
compare the performance difference when considering fringe occlusion.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main_comparison_fringe(metric, ROI, list_algorithm):
    """main function to examine the influence of fringe on HR estimation accuracy.
    Parameters
    ----------
    metric: performance evaluation metric. ['MAE', 'RMSE', 'PCC', 'CCC'].
    ROI: selected ROI for performance evaluation. 28 in total.
    list_algorithm: slected rPPG algorithm. ['LGI', 'OMIT', 'CHROM', 'POS'].
    
    Returns
    -------

    """
    # get project directory.
    dir_crt = os.getcwd()
    # loop over all selected algorithms.
    for i_algorithm in range(len(list_algorithm)):
        algorithm = list_algorithm[i_algorithm]
        # load metric dataframes. (UBFC-rPPG, UBFC-Phys).
        dir_UBFC_RPPG = os.path.join(dir_crt, 'result', 'UBFC-rPPG', 'evaluation_'+algorithm+'.csv')
        dir_UBFC_PHYS = os.path.join(dir_crt, 'result', 'UBFC-Phys', 'evaluation_'+algorithm+'.csv')
        df_UBFC_RPPG = pd.read_csv(dir_UBFC_RPPG, index_col=0)
        df_UBFC_PHYS = pd.read_csv(dir_UBFC_PHYS, index_col=0)
        # whole list.
        list_UBFC_RPPG_whole = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
        list_UBFC_PHYS_whole = list(range(1, 57))
        # finge list.
        list_UBFC_RPPG_fringe = [5, 11, 15, 30, 32, 38]
        list_UBFC_PHYS_fringe = [27, 30, 47]
        # pure list.
        list_UBFC_RPPG_pure = list(set(list_UBFC_RPPG_whole).difference(set(list_UBFC_RPPG_fringe)))
        list_UBFC_PHYS_pure = list(set(list_UBFC_PHYS_whole).difference(set(list_UBFC_PHYS_fringe)))
    
        # collect data of subjects with fringe..
        # UBFC-rPPG dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_RPPG)):
            if df_UBFC_RPPG.loc[i, 'attendant'] in list_UBFC_RPPG_fringe:
                list_idx_tmp.append(True)
            else:
                list_idx_tmp.append(False)
        df_UBFC_RPPG_fringe = df_UBFC_RPPG.loc[list_idx_tmp, :]
        df_UBFC_RPPG_fringe = df_UBFC_RPPG_fringe.reset_index()
        # UBFC-Phys dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_PHYS)):
            if df_UBFC_PHYS.loc[i, 'attendant'] in list_UBFC_PHYS_fringe:
                list_idx_tmp.append(True)
            else:
                list_idx_tmp.append(False)
        df_UBFC_PHYS_fringe = df_UBFC_PHYS.loc[list_idx_tmp, :]
        df_UBFC_PHYS_fringe = df_UBFC_PHYS_fringe.reset_index()
        list_RPPG = df_UBFC_RPPG_fringe.loc[df_UBFC_RPPG_fringe['ROI'].values==ROI, metric].values
        list_PHYS = df_UBFC_PHYS_fringe.loc[df_UBFC_PHYS_fringe['ROI'].values==ROI, metric].values
        if i_algorithm == 0:
            data_fringe = np.array(list_RPPG.tolist() + list_PHYS.tolist())
        else:
            data_fringe = data_fringe + np.array(list_RPPG.tolist() + list_PHYS.tolist())
        # collect data of subjects without fringe.
        # UBFC-rPPG dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_RPPG)):
            if df_UBFC_RPPG.loc[i, 'attendant'] in list_UBFC_RPPG_pure:
                list_idx_tmp.append(True)
            else:
                list_idx_tmp.append(False)
        df_UBFC_RPPG_pure = df_UBFC_RPPG.loc[list_idx_tmp, :]
        df_UBFC_RPPG_pure = df_UBFC_RPPG_pure.reset_index()
        # UBFC-Phys dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_PHYS)):
            if df_UBFC_PHYS.loc[i, 'attendant'] in list_UBFC_PHYS_pure:
                list_idx_tmp.append(True)
            else:
                list_idx_tmp.append(False)
        df_UBFC_PHYS_pure = df_UBFC_PHYS.loc[list_idx_tmp, :]
        df_UBFC_PHYS_pure = df_UBFC_PHYS_pure.reset_index()
        list_RPPG = df_UBFC_RPPG_pure.loc[df_UBFC_RPPG_pure['ROI'].values==ROI, metric].values
        list_PHYS = df_UBFC_PHYS_pure.loc[df_UBFC_PHYS_pure['ROI'].values==ROI, metric].values
        if i_algorithm == 0:
            data_pure = np.array(list_RPPG.tolist() + list_PHYS.tolist())
        else:
            data_pure = data_pure + np.array(list_RPPG.tolist() + list_PHYS.tolist())
    # averaging over all selected algorithms.
    data_fringe = data_fringe/len(list_algorithm)
    data_pure = data_pure/len(list_algorithm)
    # boxplot visualization.
    plt.cla()
    plt.xticks(size=17)
    plt.yticks(size=17)
    plt.boxplot(x=[data_fringe, data_pure], zorder=True, labels=['with fringe', 'without fringe'])
    dir_save = os.path.join(dir_crt, 'plot', 'occlusion', 'fringe_'+ROI+'_'+metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # write description file.
    with open(os.path.join(dir_crt, 'plot', 'occlusion', 'description_fringe.txt'), 'a') as f:
        f.writelines([metric, '\n', 
                      ROI, '\n', 
                      str(stats.levene(data_fringe, data_pure)), '\n', 
                      str(stats.ttest_ind(a=data_fringe, b=data_pure)), '\n'])


if __name__ == "__main__":
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_metric = ['MAE', 'PCC']   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_ROI = ['glabella', 'lower medial forehead', 'left lower lateral forehead', 'right lower lateral forehead']   # 28 facial ROIs.
    # create the description file.
    with open(os.path.join(os.getcwd(), 'plot', 'motion', 'description_fringe.txt'), 'w') as f:
        pass
    # loop over all metrics.
    for metric in list_metric:
        # loop over all selected ROIs.
        for ROI in list_ROI:
            print([metric, ROI])
            main_comparison_fringe(metric=metric, ROI=ROI, list_algorithm=list_algorithm)