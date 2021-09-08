import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
def plot_variable_pairs(df):
    for col in df.columns:
        if df[col].dtype == 'O':
            sns.pairplot(data=df, hue=col, corner=True)
            plt.show()


def months_to_years(df, tenure):
    """
    Takes unscaled telco churn dataframe and returns a dataframe with a new feature tenure_years
    """
    df['tenure_years'] = round(df['tenure'] / 12, 0).astype(int)
    return df


def plot_categorical_and_continuous_vars(train, cats, quants):
    """
    Function takes in a df, categorical variables, and quantitative variable and returns a violin plot, 
    ked plot, and box plot for each combination of categorical vars to quantitative vars.
    """
    for quant in train[quants]:
        print('-------------------------------------------------')
        print(f'{quant}')
        print('-------------------------------------------------')
        for cat in train[cats]:
            print(f'vs. {cat}')    
            fig = plt.figure(figsize=(20,4))

            #subplot 1
            plt.subplot(131)
            plt.title('Violin Plot')
            sns.violinplot(x=quant, y=cat, data=train, palette="Set2")

            #subplot 2
            plt.subplot(132)
            plt.title('Kernel Density Estimate Plot')
            sns.kdeplot(data=train, x=quant, hue=cat, multiple="stack", shade_lowest=True, palette="Set2")

            #subplot 3
            plt.subplot('133')
            sns.boxenplot(data=train, x=quant, y=cat, palette="Set2")
            plt.title('Box Plot')
            print('---------------')
            plt.show()