"""
the experiment for the correlation analysis between skin thickness and ROI performance.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def main_corr_skin(metric, skin, list_algorithm):
    """main function for correlation analysis of epidermal thickness and rPPG performance metric.
    Parameters
    ----------
    metric: performance evaluation metric. ['MAE', 'RMSE', 'PCC', 'CCC'].
    skin: type of skin tissue. ['dermal', 'epidermal'].
    list_algorithm: slected rPPG algorithm. ['LGI', 'OMIT', 'CHROM', 'POS'].
    
    Returns
    -------

    """
    # get project directory.
    dir_crt = os.getcwd()
    data_evaluation_avg = np.zeros(shape=[28])
    for algorithm in list_algorithm:
        # get evaluation data.
        # UBFC-rPPG dataset.
        dir_evaluation_UBFC_RPPG = os.path.join(dir_crt, 'result', 'UBFC-rPPG', 'evaluation_' + algorithm + '.csv')
        df_evaluation_UBFC_RPPG = pd.read_csv(dir_evaluation_UBFC_RPPG, index_col=0)
        df_evaluation_UBFC_RPPG = df_evaluation_UBFC_RPPG.reset_index()
        # evaluation data initialization.
        data_evaluation = []
        for roi_tmp in np.unique(df_evaluation_UBFC_RPPG['ROI'].values).tolist():
            # UBFC-rPPG + UBFC-Phys.
            data_evaluation.append(df_evaluation_UBFC_RPPG.loc[df_evaluation_UBFC_RPPG['ROI'].values==roi_tmp, metric].mean())
        data_evaluation = np.array(data_evaluation)
        data_evaluation_avg = data_evaluation_avg + data_evaluation
    data_evaluation_avg = data_evaluation_avg/len(list_algorithm)
    # skin thickness data.
    dir_anatomy = os.path.join(dir_crt, 'data', 'anatomy.csv')
    df_anatomy = pd.read_csv(dir_anatomy, index_col=0)
    df_anatomy = df_anatomy.reset_index()
    data_skin = []
    for roi_tmp in np.unique(df_evaluation_UBFC_RPPG['ROI'].values).tolist():
        data_skin.append(df_anatomy.loc[df_anatomy['ROI'].values==roi_tmp, skin+'_thickness'].mean())
    data_skin = np.array(data_skin)
    # visualization.
    plt.cla()
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    sns.regplot(x=skin, 
                y=metric, 
                data=pd.DataFrame(data=np.stack([data_skin, data_evaluation_avg], axis=1), 
                                  columns=[skin, metric]),
                scatter_kws={"color": "purple"}, 
                line_kws={"color": "black"},
                label=None
                )
    dir_save = os.path.join(dir_crt, 'plot', 'corr', skin+'_'+metric+'.png')
    plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    # correlation analysis.
    X = sm.add_constant(data_skin)
    model = sm.OLS(data_evaluation_avg, X)
    result = model.fit()
    # write description file.
    with open(os.path.join(dir_crt, 'plot', 'corr', 'description_skin.txt'), 'a') as f:
        f.writelines([skin, '\n', 
                      metric, '\n', 
                      'Pearson Correlation Coefficient: '+str(np.corrcoef(data_skin, data_evaluation_avg)[0, 1]), '\n', 
                      'F-statistic: '+str(result.fvalue), '\n',
                      'F-pvalue: '+str(result.f_pvalue), '\n'])


if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_metric = ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW']   # ['MAE', 'RMSE', 'PCC', 'CCC', 'DTW'].
    list_skin = ['dermal', 'epidermal']   # ['dermal', 'epidermal'].
    # create the description file.
    with open(os.path.join(os.getcwd(), 'plot', 'corr', 'description_skin.txt'), 'w') as f:
        pass
    # loop over all selected evaluation metrics.
    for metric in list_metric:
        # loop over all selected skin tissue.
        for skin in list_skin:
            print([metric, skin])
            main_corr_skin(metric=metric, skin=skin, list_algorithm=list_algorithm)