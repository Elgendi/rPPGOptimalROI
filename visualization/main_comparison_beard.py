"""
compare the performance difference when considering beard occlusion.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def main_comparison_beard(metric, ROI, list_algorithm):
    """main function to examine the influence of beard on HR estimation accuracy.
    Parameters
    ----------
    metric: performance evaluation metric. ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    ROI: selected ROI for performance evaluation. 28 in total.
    list_algorithm: list of slected rPPG algorithms. ['LGI', 'OMIT', 'CHROM', 'POS'].
    
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
        # beard list.
        list_UBFC_RPPG_beard = [3, 12, 13, 14, 23, 24, 30, 31, 32, 39, 44]
        list_UBFC_PHYS_beard = [2, 19, 38, 48, 49]
        # pure list.
        list_UBFC_RPPG_pure = list(set(list_UBFC_RPPG_whole).difference(set(list_UBFC_RPPG_beard)))
        list_UBFC_PHYS_pure = list(set(list_UBFC_PHYS_whole).difference(set(list_UBFC_PHYS_beard)))
    
        # collect data of subjects with beard.
        # UBFC-rPPG dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_RPPG)):
            if df_UBFC_RPPG.loc[i, 'attendant'] in list_UBFC_RPPG_beard:
                list_idx_tmp.append(True)
            else:
                list_idx_tmp.append(False)
        df_UBFC_RPPG_beard = df_UBFC_RPPG.loc[list_idx_tmp, :]
        df_UBFC_RPPG_beard = df_UBFC_RPPG_beard.reset_index()
        # UBFC-Phys dataset.
        list_idx_tmp = []
        for i in range(len(df_UBFC_PHYS)):
            if df_UBFC_PHYS.loc[i, 'attendant'] in list_UBFC_PHYS_beard:
                list_idx_tmp.append(True)
            else:
                list_idx_tmp.append(False)
        df_UBFC_PHYS_beard = df_UBFC_PHYS.loc[list_idx_tmp, :]
        df_UBFC_PHYS_beard = df_UBFC_PHYS_beard.reset_index()
        list_RPPG = df_UBFC_RPPG_beard.loc[df_UBFC_RPPG_beard['ROI'].values==ROI, metric].values
        list_PHYS = df_UBFC_PHYS_beard.loc[df_UBFC_PHYS_beard['ROI'].values==ROI, metric].values
        if i_algorithm == 0:
            data_beard = np.array(list_RPPG.tolist() + list_PHYS.tolist())
        else:
            data_beard = data_beard + np.array(list_RPPG.tolist() + list_PHYS.tolist())
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
    data_beard = data_beard/len(list_algorithm)
    data_pure = data_pure/len(list_algorithm)
    # boxplot visualization.
    plt.cla()
    plt.xticks(size=17)
    plt.yticks(size=17)
    plt.boxplot(x=[data_beard, data_pure], zorder=True, labels=['with beard', 'without beard'])
    dir_save = os.path.join(dir_crt, 'plot', 'occlusion', 'beard_'+ROI+'_'+metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # write description file.
    with open(os.path.join(dir_crt, 'plot', 'occlusion', 'description_beard.txt'), 'a') as f:
        f.writelines([metric, '\n', 
                      ROI, '\n', 
                      str(stats.levene(data_beard, data_pure)), '\n', 
                      str(stats.ttest_ind(a=data_beard, b=data_pure)), '\n'])


if __name__ == "__main__":
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_metric = ['MAE', 'PCC']   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_ROI = ['philtrum', 'chin', 'right upper lip', 'left upper lip']   # 28 facial ROIs.
    # create the description file.
    with open(os.path.join(os.getcwd(), 'plot', 'occlusion', 'description_beard.txt'), 'w') as f:
        pass
    # loop over all metrics.
    for metric in list_metric:
        # loop over all selected ROIs.
        for ROI in list_ROI:
            print([metric, ROI])
            main_comparison_beard(metric=metric, ROI=ROI, list_algorithm=list_algorithm)