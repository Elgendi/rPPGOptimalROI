"""
The experiment for the correlation analysis between facial surface orientation and ROI performance.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def main_corr_so(metric, list_algorithm):
    """main function to evaluate the effect of facial surface orientation on ROI performance.
    Parameters
    ----------
    metric: selecte evaluation metric.   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_algorithm: list of selecte rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].

    Returns
    -------

    """
    # get current directory.
    dir_crt = os.getcwd()
    # data initialization.
    data_evaluation_avg = np.zeros(shape=[28])
    # loop over all selected rPPG algorithms.
    for algorithm in list_algorithm:
        # get evaluation data.
        dir_evaluation = os.path.join(dir_crt, 'result', 'UBFC-rPPG', 'evaluation_' + algorithm + '.csv')
        df_evaluation = pd.read_csv(dir_evaluation, index_col=0)
        df_evaluation = df_evaluation.reset_index()
        data_evaluation = []
        # loop over all facial ROIs.
        for roi_tmp in np.unique(df_evaluation['ROI'].values).tolist():
            data_evaluation.append(df_evaluation.loc[df_evaluation['ROI'].values==roi_tmp, metric].mean())
        data_evaluation = np.array(data_evaluation)
        data_evaluation_avg = data_evaluation_avg + data_evaluation
    # take average over all selected rPPG algorithms.
    data_evaluation_avg = data_evaluation_avg/len(list_algorithm)
    # number of pixels.
    list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
    # data initialization.
    data_so_avg = np.zeros(shape=[28])
    for i_attendant in list_attendant:
        data_so = []
        dir_pixel = os.path.join(dir_crt, 'data', 'UBFC-rPPG', 'feature', str(i_attendant)+'.csv')
        df_pixel = pd.read_csv(dir_pixel, index_col=None)
        for roi_tmp in np.unique(df_pixel['ROI'].values).tolist():
            data_so.append(df_pixel.loc[df_pixel['ROI'].values==roi_tmp, 'median_so'].mean())
        data_so_avg = data_so_avg + np.array(data_so)
    # take average over all selected rPPG algorithms.
    data_so_avg = data_so_avg/len(list_attendant)
    # visualization and save data.
    plt.cla()
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    sns.regplot(x='median_so', 
                y=metric, 
                data=pd.DataFrame(data=np.stack([data_so_avg, data_evaluation_avg], axis=1), 
                                  columns=['median_so', metric]),
                scatter_kws={"color": "purple"}, 
                line_kws={"color": "black"},
                label=None
                )
    dir_save = os.path.join(dir_crt, 'plot', 'corr', 'so_'+metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # regression analysis.
    X = sm.add_constant(data_so_avg)
    model = sm.OLS(data_evaluation_avg, X)
    result = model.fit()
    # write description file.
    with open(os.path.join(dir_crt, 'plot', 'corr', 'description_so.txt'), 'a') as f:
        f.writelines([metric, '\n', 
                      'Pearson Correlation Coefficient: '+str(np.corrcoef(data_so_avg, data_evaluation_avg)[0, 1]), '\n', 
                      'F-statistic: '+str(result.fvalue), '\n', 
                      'F-pvalue: '+str(result.f_pvalue), '\n'])

if __name__ == "__main__":
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']
    list_metric = ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW']
    # create the description file.
    with open(os.path.join(os.getcwd(), 'plot', 'corr', 'description_so.txt'), 'w') as f:
        pass
    # loop over all selected evaluation metrics.
    for metric in list_metric:
        print([metric])
        main_corr_so(metric=metric, list_algorithm=list_algorithm)