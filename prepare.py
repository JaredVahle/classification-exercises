import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")

import acquire

df = acquire.get_titanic_data()

def clean_data(df):
    '''
    This function will drop any duplicate observations, 
    drop columns not needed, fill missing embarktown with 'Southampton'
    and create dummy vars of sex and embark_town. 
    '''
    df.drop_duplicates(inplace=True)
    df.drop(columns=['deck', 'embarked', 'class', 'age'], inplace=True)
    df.embark_town.fillna(value='Southampton', inplace=True)
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    return pd.concat([df, dummy_df], axis=1)

def split_titanic_data(df):
    """
    splits the data in train validate and test 
    """
    train, test = train_test_split(df, test_size = 0.2, random_state = 123, stratify = df.survived)
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train.survived)
    
    return train, validate, test

def impute_mode(train, validate, test):
    '''
    impute mode for embark_town
    '''
    imputer = SimpleImputer(strategy='most_frequent')
    train[['embark_town']] = imputer.fit_transform(train[['embark_town']])
    validate[['embark_town']] = imputer.transform(validate[['embark_town']])
    test[['embark_town']] = imputer.transform(test[['embark_town']])
    return train, validate, test

def prep_titanic_data(df):
    """
    takes in a data from titanic database, cleans the data, splits the data
    in train validate test and imputes the missing values for embark_town. 
    Returns three dataframes train, validate and test.
    """
    df = clean_data(df)
    train, validate, test = split_titanic_data(df)
    train, validate, test = impute_mode(train, validate, test)
    return train, validate, test

def clean_iris(df):
    df.drop_duplicates(inplace = True)
    df.rename(columns = {"species_name":"species"}, inplace = True)
    df_dummy = pd.get_dummies(df[["species"]],drop_first = True)
    return pd.concat([df, df_dummy], axis=1)

def split_iris_data(df):
    """
    splits the data in train validate and test 
    """
    train, test = train_test_split(df, test_size = .2, random_state = 50, stratify = df.species)
    train, validate = train_test_split(train, test_size=.25, random_state=50, stratify = train.species)
    
    return train, validate, test

def prep_iris_data(df):
    """
    takes in a data from iris database, cleans the data, splits the data
    in train validate test. 
    Returns three dataframes train, validate and test.   
    """
    df = clean_iris(df)
    train, validate, test = split_iris_data(df)
    return train, validate, test