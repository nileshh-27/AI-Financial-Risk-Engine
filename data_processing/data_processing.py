import pandas as pd

df = pd.read_csv('/media/gowtham/MULTIMEDIA/Projects/ai-financial-risk-engine/AI-Financial-Risk-Engine/synthetic_financial_risk_dataset.csv')
df1 = pd.read_csv('/media/gowtham/MULTIMEDIA/Projects/ai-financial-risk-engine/AI-Financial-Risk-Engine/synthetic_financial_risk_dataset1.csv')
print(df.head())
print(df1.head())

df4 =pd.concat([df1 , df])
df4 = df.drop(columns=['customer_id'])
df4.to_csv('synthetic_financial_dataset_2k_datapoints.csv' , index = False)