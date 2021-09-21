import os
from env import host, user, password
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing


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

def minmax_scaler(X_train, X_validate, X_test, quants):
    """
    Takes in split data and individually scales each features to a value within the range of 0 and 1.
    Uses min-max scaler method from sklearn.
    Returns datasets with new, scaled columns to the added.
    """
    # Scale the data
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit the scaler
    scaler.fit(X_train[quants])

    # Use the scaler to transform train, validate, test
    X_train_scaled = scaler.transform(X_train[quants])
    X_validate_scaled = scaler.transform(X_validate[quants])
    X_test_scaled = scaler.transform(X_test[quants])


    # Turn everything into a dataframe
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train[quants].columns)
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=X_train[quants].columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train[quants].columns)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled



################### Cleans df ###################

def clean_zillow(df):
    """
    Function that cleans:
        - drops any rows with null values, 
        - drops any duplicate rows, 
        - converts columns to correct datatypes,
        - edits 'fips' column to proper zipcode format
        - renames all columns
    """
    
    # Drops duplicated id rows
    df.drop_duplicates(subset=['parcelid'], inplace=True)
    
    # Drops duplicate target variable
    df.dropna(subset=['taxvaluedollarcnt', 'taxamount'], inplace=True)
        
    # drops unusable columns    
    df.drop(columns=['finishedfloor1squarefeet', 'id', 'id', 'airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid', 'assessmentyear', 'censustractandblock', 'structuretaxvaluedollarcnt', 'buildingqualitytypeid', 'latitude', 'longitude', 'regionidcounty', 'regionidzip', 'regionidcity', 'regionidneighborhood', 'rawcensustractandblock', 'storytypeid', 'heatingorsystemtypeid', 'typeconstructiontypeid', 'unitcnt', 'numberofstories', 'landtaxvaluedollarcnt', 'yardbuildingsqft26', 'taxdelinquencyflag', 'calculatedbathnbr', 'pooltypeid10', 'finishedsquarefeet50', 'finishedsquarefeet15', 'finishedsquarefeet13', 'basementsqft', 'decktypeid', 'finishedsquarefeet6', 'yardbuildingsqft17', 'poolsizesum', 'taxdelinquencyyear', 'hashottuborspa', 'id.1', 'finishedsquarefeet12', 'propertyzoningdesc', 'propertycountylandusecode', 'transactiondate', 'propertylandusedesc', 'propertylandusetypeid', 'parcelid', 'pooltypeid7', 'pooltypeid2', 'garagetotalsqft'], inplace=True)
    
    # Renames columns
    df.rename(columns={"bathroomcnt": "bath",
          "bedroomcnt": "bed",
          "calculatedfinishedsquarefeet": "sqft",
          "fullbathcnt": "full_bath",
          "poolcnt": "pool",
          "roomcnt": "rooms",    
          "threequarterbathnbr": "three_qtr_bath",
          "yearbuilt": "year",
          "lotsizesquarefeet": "lot_sqft",
          "taxvaluedollarcnt": "tax_value",
          "fireplacecnt": "fireplaces"
          }, inplace=True)
    
    # Imputes missing values
    df.fireplaces[list(df['fireplaces'][df.fireplaceflag==1].index.values)] = df.fireplaces[list(df['fireplaces'][df.fireplaceflag==1].index.values)].fillna(1)
    df.fireplaces.fillna(0, inplace=True)
    df['pool'].fillna(value=0, inplace=True)
    df['garagecarcnt'].fillna(value=0, inplace=True)
    df['full_bath'].fillna(value=0, inplace=True)
    df['three_qtr_bath'].fillna(value=0, inplace=True)
    mean_vals = df[['sqft', 'lot_sqft', 'year']].mean()
    df[['sqft', 'lot_sqft', 'year']] = df[['sqft', 'lot_sqft', 'year']].fillna(mean_vals)
    
    # Resets index and converts data types
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)
    df['bed'] = df.bed.astype(int)
    df['fips'] = df.fips.astype(int)
    df['full_bath'] = df.full_bath.astype(int)
    df['garagecarcnt'] = df.garagecarcnt.astype(int)
    df['pool'] = df['pool'].astype(int)
    df['rooms'] = df.rooms.astype(int)
    df['three_qtr_bath'] = df.three_qtr_bath.astype(int)
    df['year'] = round(df.year, 0).astype(int)
    df['fireplaces'] = df.fireplaces.astype(int)
    df.drop(columns=['full_bath', 'fireplaceflag', 'logerror'], inplace=True)
    
    # Removes outliers of target
    df = df[df.tax_value < df.tax_value.mean() + 3 * df.tax_value.std()]
    df = df[df.taxamount < df.taxamount.mean() + 3 * df.taxamount.std()]
    
    return df                                                       

                                                                
################### X, y split Funciton ###################

def X_y_split(train, validate, test, target):
    """
    Functions that takes in trainm validate, test, and target var and split to X and y datasets
    """
    # Setup X and y
    X_train = train.drop(columns=target)
    y_train = train[target]

    X_validate = validate.drop(columns=target)
    y_validate = validate[target]

    X_test = test.drop(columns=target)
    y_test = test[target]
    return X_train, y_train, X_validate, y_validate, X_test, y_test
