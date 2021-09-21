import os
from env import host, user, password
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.impute import SimpleImputer


################### Connects to Sequel Ace using credentials ###################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


################### Create new dataframe from SQL db ###################
    
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''

    # Create SQL query.
    sql_query = """
           SELECT bedroomcnt, 
               bathroomcnt, 
               calculatedfinishedsquarefeet, 
               taxvaluedollarcnt, 
               yearbuilt, 
               taxamount, 
               fips
            FROM properties_2017
            WHERE propertylandusetypeid = 261;            
            """
    # Read in DataFrame from Codeup's SQL db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df


################### Acquire existing csv file ###################

def get_zillow_data():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df


################### Cleans df ###################

def clean_zillow(df):
    """
    Function that cleans:
        - dropps any rows with null values, 
        - drops any duplicate rows, 
        - converts columns to correct datatypes,
        - edits 'fips' column to proper zipcode format
        - renames all columns
    """
    
    # Drops rows with null values for target variable
    df.dropna(subset=['taxvaluedollarcnt'], inplace=True)
    
    # Drop duplicates rows
    df.drop_duplicates(inplace=True)
        
    # impute mean values    
    mean_values = df.mean(axis=0)
    train_df_new = df.fillna(mean_values, inplace=True)
    
    # Changes datatypes
    df['bedroomcnt'] = df.bedroomcnt.astype(int)  # from float to int
    df['yearbuilt'] = df.yearbuilt.astype(int)  # from float to int
    df['fips'] = ('0' + df.fips.astype(str)).astype(float)  # changes fips to int, then string, then adds '0' to front
    
    # Rename columns
    df = df.rename(columns={'bedroomcnt': 'bed',
                       'bathroomcnt': 'bath',
                       'calculatedfinishedsquarefeet': 'sqft',
                       'taxvaluedollarcnt': 'prop_value',
                       'yearbuilt': 'year',
                       'taxamount': 'prop_tax',
                       'fips': 'zip'})
    return df


################### Impute mean values ###################

def imp_mean(train, validate, test):
    # Create the SimpleImputer object
    imputer = SimpleImputer(missing_values = None, strategy='mean')
    
    # Fit the imputer to the columns in the training df
    imputer = imputer.fit(train[['bed','bath','sqft','year','prop_tax']])
          
    # Transform the data
    train[['bed','bath','sqft','year','prop_tax']] = imputer.transform(train[['bed','bath','sqft','year','prop_tax']])
    validate[['bed','bath','sqft','year','prop_tax']] = imputer.transform(validate[['bed','bath','sqft','year','prop_tax']])
    test[['bed','bath','sqft','year','prop_tax']] = imputer.transform(test[['bed','bath','sqft','year','prop_tax']])
    return train, validate, test                                                        

                                                                 
################### Remove outliers ###################

def remove_outliers(df):
    """
    Takes in df and removes outliers that are greater than the upper bound (>Q3) or less than the lower bound (<Q1)
    """
    # Iterates through columns (except 'zip')
    for col in df.columns.drop('zip'):
        # Creates !st and 3rd quartile vars
        Q1, Q3 = df[col].quantile([.25, .75])
        # Creates IQR var
        IQR = Q3 - Q1
        #Creates Upper and Lower vars
        UB = Q3 + 1.5 * IQR
        LB = Q1 - 1.5 * IQR
        # drops rows with column data greater than 
        df = df[(df[col] <= UB) & (df[col] >= LB)]
    return df


################### Split the data ###################

def split_data(df):
    """
    Splits the data into train, validate, and test dataframes each comprised of 72%, 18%, and 10% of the original dataframe, respectively.
    """
    train_validate, test = train_test_split(df, 
                                            test_size=.1, 
                                            random_state=123)
    train, validate = train_test_split(train_validate, 
                                        test_size=.2, 
                                        random_state=123)
    return train, validate, test


################### Min-max Scaling ###################

def minmax_scaler(train, validate, test):
    """
    Takes in split data and individually scales each features to a value within the range of 0 and 1.
    Uses min-max scaler method from sklearn.
    Returns datasets with new, scaled columns to the added.
    """
    x = []
    minmax_train = train.copy()
    minmax_validate = validate.copy()
    minmax_test = test.copy()
    for col in train.columns:
        # 1. create the object
        mm_scaler = sklearn.preprocessing.MinMaxScaler()

        # 2. fit the object (learn the min and max value)
        mm_scaler.fit(train)

        # 3. use the object (use the min, max to do the transformation)
        scaled_zillow_train = mm_scaler.transform(train)
        scaled_zillow_validate = mm_scaler.transform(validate)
        scaled_zillow_test = mm_scaler.transform(test)

        x.append(col + '_scaled')

    # assign the scaled values as new columns in the datasets
    minmax_train[x] = scaled_zillow_train
    minmax_train = minmax_train[minmax_train.columns[7:]]
    
    minmax_validate[x] = scaled_zillow_validate
    minmax_validate = minmax_validate[minmax_validate.columns[7:]]
    
    minmax_test[x] = scaled_zillow_test
    minmax_test = minmax_test[minmax_test.columns[7:]]
    return minmax_train, minmax_validate, minmax_test


################### X, y split Funciton ###################

def Xy_split(train, validate, test, minmax_train, minmax_validate, minmax_test):
    X_exp, y_train, y_validate, y_test = train.drop(columns='prop_value'), train.prop_value, validate.prop_value, test.prop_value
    X_train, X_validate, x_test = minmax_train, minmax_validate, minmax_test
    return X_exp, X_train, X_validate, x_test, y_train, y_validate, y_test


################### Wrangle Funciton ###################

def wrangle_zillow():
    """
    Automates all functions contained within module (Unscaled)
    """
    df = get_zillow_data()
    df = clean_zillow(df)
    df = remove_outliers(df)  
    train, validate, test = split_data(df)
    #imp_train, imp_validate, imp_test = imp_mean(train, validate, test)                                                         
    minmax_train, minmax_validate, minmax_test = minmax_scaler(train, validate, test)
    X_exp, X_train, X_validate, x_test, y_train, y_validate, y_test = Xy_split(train, validate, test, minmax_train, minmax_validate, minmax_test)
    return df, X_exp, X_train, X_validate, x_test, y_train, y_validate, y_test


################### Split Scaled and Target ###################
