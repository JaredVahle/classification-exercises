import env
import os
import pandas as pd

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'



def get_telco_data():
    filename = "telco_churn.csv"

    if os.path.isfile("telco_churn.csv"):
        return pd.read_csv("telco_churn.csv")
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''
SELECT *
FROM customers
JOIN contract_types ON contract_types.contract_type_id = customers.contract_type_id
JOIN internet_service_types ON internet_service_types.internet_service_type_id = customers.internet_service_type_id
JOIN payment_types ON payment_types.payment_type_id = customers.payment_type_id;
''', get_connection("telco_churn"))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv("telco_churn.csv")

        # Return the dataframe to the calling code
        return df 