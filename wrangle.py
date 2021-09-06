import os
from env import host, user, password
import numpy as np
import pandas as pd


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
    """
    
    # Drops null values
    df.dropna(axis=0, inplace=True)
    
    # Drops duplicates rows
    df.drop_duplicates(inplace=True)
    
    # Changes datatypes
    df['bedroomcnt'] = df.bedroomcnt.astype(int)  # from float to int
    df['yearbuilt'] = df.yearbuilt.astype(int)  # from float to int
    df['fips'] = '0' + df.fips.astype(int).astype(str)  # changes fips to int, then string, then adds '0' to front
    
    return df