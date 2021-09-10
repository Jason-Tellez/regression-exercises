import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_residuals(y, yhat):
    sns.scatterplot(x=y, y=yhat)
    plt.axhline(yhat.mean(), ls = ':')
    plt.show()



def regression_errors(y, yhat):
    residual = yhat - y
    residual_sq = residual**2
    SSE = residual_sq.sum()
    ESS = ((yhat - y.mean())**2).sum()
    TSS = SSE + ESS
    MSE = SSE/len(y)
    RMSE = MSE**0.5
    return SSE, ESS, TSS, MSE, RMSE



def baseline_mean_errors(y):
    baseline = y.mean()
    base_res = y - baseline
    base_res_sq = base_res**2
    SSE_baseline = base_res_sq.sum()
    MSE_baseline = SSE_baseline/len(y)
    RMSE_baseline = MSE_baseline**0.5
    return SSE_baseline, MSE_baseline, RMSE_baseline



def better_than_baseline(y, yhat):
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    return RMSE < RMSE_baseline