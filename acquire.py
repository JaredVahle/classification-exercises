# where im going to put my finished functions.
import env
import os
import pandas as pd


def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile("titanic.csv"):
        return pd.read_csv("titanic.csv")
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv("titanic.csv")

        # Return the dataframe to the calling code
        return df 

def get_iris_data():
    filename = "iris_db.csv"

    if os.path.isfile("iris_db.csv"):
        return pd.read_csv("iris_db.csv")
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''
SELECT *
FROM species
JOIN measurements ON measurements.species_id = species.species_id;
''', get_connection('iris_db'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv("iris_db.csv")

        # Return the dataframe to the calling code
        return df