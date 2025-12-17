import pandas as pd
df = pd.read_excel("/Users/danyukezz/Desktop/3 year 1 semester/MLOps/mlops_exam/got_mltable/got_persona_dataset_100.xlsx")   # openpyxl used under the hood
df.to_csv("/Users/danyukezz/Desktop/3 year 1 semester/MLOps/mlops_exam/got_mltable/got_persona.csv", index=False)
