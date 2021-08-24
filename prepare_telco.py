def clean_telco(df):
    df.total_charges.replace(to_replace = {" ":"0"}, inplace = True)
    df.total_charges = df.total_charges.astype("float")
    df["auto_pay"] = df.payment_type.str.contains("auto")
    df["auto_pay"] = df.auto_pay.replace(to_replace = [True,False],value = [1,0])
    dummy_df = pd.get_dummies(df[['gender', 'payment_type',"internet_service_type","contract_type"]])
    dummy_df.columns = [col.lower().replace(" ","_") for col in dummy_df]
    
    df["partner"] = df.partner.replace(to_replace = ["Yes","No"],value = [1,0])
    df["dependents"] = df.dependents.replace(to_replace = ["Yes","No"],value = [1,0])
    df["churn"] = df.churn.replace(to_replace = ["Yes","No"],value = [1,0])
    df["multiple_lines"] = df.paperless_billing.replace(to_replace = ["Yes","No","No phone service"],value = [1,0,0])
    df["paperless_billing"] = df.paperless_billing.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["tech_support"] = df.tech_support.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["device_protection"] = df.device_protection.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["online_backup"] = df.online_backup.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["online_security"] = df.online_security.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["streaming_tv"] = df.streaming_tv.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["streaming_movies"] = df.streaming_movies.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["phone_service"] = df.phone_service.replace(to_replace = ["Yes","No","No internet service"],value = [1,0,0])
    df["multiple_lines"] = df.multiple_lines.replace(to_replace = ["Yes","No","No phone service"],value = [1,0,0])
    df["add_ons"] =  df.online_security + df.online_backup + df.device_protection + df.tech_support + df.streaming_tv + df.streaming_movies
    df.drop(columns = ["customer_id",
                       "gender",
                       "payment_type",
                       "internet_service_type",
                       "contract_type",
                   'internet_service_type_id',
                   "contract_type_id",
                   "payment_type_id",
                  "contract_type_id.1",
                   "internet_service_type_id.1",
                   "payment_type_id.1",
                   "Unnamed: 0"],inplace = True)
    return pd.concat([df, dummy_df], axis=1)
    