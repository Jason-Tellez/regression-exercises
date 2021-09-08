import os
from env import host, user, password
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing


#######################################################
###################     Acquire     ###################
#######################################################


################### Connects to Sequel Ace using credentials ###################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
################### Create new dataframe from SQL db ###################
    
def new_telco_data():
    '''
    This function reads the telco data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''

    # Create SQL query.
    sql_query = """
           SELECT c.customer_id,
                c.gender, 
                c.senior_citizen,
                c.partner,
                c.dependents,
                c.tenure,
                c.phone_service,
                c.multiple_lines,
                c.online_security,
                c.device_protection,
                c.tech_support,
                c.streaming_tv,
                c.streaming_movies,
                c.paperless_billing,
                c.monthly_charges,
                c.total_charges,
                c.churn,
                ct.contract_type,
                i.internet_service_type,
                p.payment_type
            FROM customers as c
            JOIN contract_types as ct USING (contract_type_id)
            JOIN internet_service_types as i USING (internet_service_type_id)
            JOIN payment_types as p USING (payment_type_id);
                """
    
    # Read in DataFrame from Codeup's SQL db.
    df = pd.read_sql(sql_query, get_connection('telco_churn'))
    
    return df


################### Acquire existing csv file ###################

def get_telco_data():
    '''
    This function reads in telco data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('telco_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_telco_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_df.csv')
        
    return df



#######################################################
###################     Prepare     ###################
#######################################################


################### Prepare dataframe by editing columns ###################

def  prep_data(df):
    '''
    This function prepares and cleans the data by:
        - drops rows with empty total_charges values
        - repeats same dummy var process with deleting any new columns
            > combines dummy columns with original df
        - drops any existing duplicate rows
        - drops unsuable or unnecessary columns
        - rename several columns
    '''
    # replace empty spaces with 0.00 and convert total_charges to float dtype
    df.drop(index=df[df.total_charges==' '].index.values.tolist(), inplace=True)
    df['total_charges'] = df.total_charges.astype(float)

    # creates dummy vars for gender and payment_type, drops first new var, concats dummy vars with original df
    #dummy_df = pd.get_dummies(df['gender'], dummy_na=False, drop_first=True) 
    #df = pd.concat([df, dummy_df], axis=1)

    # creates dummy vars for internet_service_type and contract_type then concats dummy vars with original df
    #dummy_df = pd.get_dummies(df[['internet_service_type', 'contract_type', 'payment_type']], dummy_na=False)
    #df = pd.concat([df, dummy_df], axis=1)

    # drops unusable or unecessary columns and columns with new dummy vars
    df.drop(columns=['customer_id'], inplace=True)
    
    # adds tenure years column
    #df['tenure_years'] = round(df.tenure / 12, 1)
    
    # renames columns
    df.rename(columns={'phone_service': 'has_phone',
                     'online_security': 'online_sec',
                     'device_protection': 'dev_prot',
                     'streaming_tv': 'stream_tv',
                     'streaming_movies': 'stream_mov',
                     'paperless_billing': 'paperless',
                     #'payment_type_Credit card (automatic)': 'credit_card',
                     #'payment_type_Electronic check': 'elec_check',
                     #'payment_type_Mailed check': 'mail_check',
                     #'internet_service_type_DSL': 'has_DSL',
                     #'internet_service_type_Fiber optic': 'has_Fiber',
                     #'internet_service_type_None': 'no_internet',
                     #'contract_type_One year': 'one_year',
                     #'contract_type_Month-to-month': 'm2m',
                     #'contract_type_Two year': 'two_year',
                     #'payment_type_Bank transfer (automatic)': 'bank_trans',
                    }, inplace=True)
    
    # Change 0,1 to 'Yes'/'No' for categorical columns
    for col in df.columns:
        if (df[col].nunique() == 2) and (df[col].dtype != 'O'):
            df.replace({col: {1: 'Yes', 0: 'No'}}, inplace=True)

    return df


################### Train, validate, and test ###################

def train_validate_test_split(df, target='churn', seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test


################### Min-Max Scaler Funciton ###################

def minmax_scaler(train, validate, test):
    """
    Takes in split data and individually scales each features to a value within the range of 0 and 1.
    Uses min-max scaler method from sklearn.
    Returns datasets with new, scaled columns to the added.
    """
    x = []
    # create categorical and quantitative vars to separate columns
    quants = ['monthly_charges','total_charges','tenure']
    cats = list(train.columns[~train.columns.isin(quants)])
    minmax_train = train[cats].copy()
    minmax_validate = validate[cats].copy()
    minmax_test = test[cats].copy()
    for col in train[quants]:
        # 1. create the object
        mm_scaler = sklearn.preprocessing.MinMaxScaler()

        # 2. fit the object (learn the min and max value)
        mm_scaler.fit(train[quants])

        # 3. use the object (use the min, max to do the transformation)
        scaled_telco_train = mm_scaler.transform(train[quants])
        scaled_telco_validate = mm_scaler.transform(validate[quants])
        scaled_telco_test = mm_scaler.transform(test[quants])

        x.append(col)

    # assign the scaled values as new columns in the datasets
    minmax_train[x] = scaled_telco_train
    minmax_validate[x] = scaled_telco_validate
    minmax_test[x] = scaled_telco_test
    
    return minmax_train, minmax_validate, minmax_test


################### Wrangle Function ###################

def wrangle_telco():
    """
    Initiates all functions with scaling
    """
    df = get_telco_data()
    df = prep_data(df)
    train, validate, test = train_validate_test_split(df, target='churn', seed=123)
    minmax_train, minmax_validate, minmax_test = minmax_scaler(train, validate, test)

    return minmax_train, minmax_validate, minmax_test

def wrangle_unscaled_telco():
    """
    Initiates all functions without scaling
    """
    df = get_telco_data()
    df = prep_data(df)
    train, validate, test = train_validate_test_split(df, target='churn', seed=123)

    return train, validate, test