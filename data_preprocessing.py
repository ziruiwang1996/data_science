import pandas as pd
df = pd.read_csv('hw1_data.csv')
#Combine ‘date’, ‘month’ and ‘year’ into a single attribute.
df['Date'] = pd.to_datetime(df[['Month', 'Day', 'Year']])
#Remove redundant attributes (all the same values) and meaningless attributes (all val- ues are different).
df = df.drop(columns=['PROJ_PERIOD_ID', 'valid', 'obs_id', 'Month', 'Day', 'Year'])
#Convert all categorical attributes to quoted numbers.
df['reviewed'] = df['reviewed'].astype(str)
df['day1_am'] = df['day1_am'].astype(str)
df['day1_pm'] = df['day1_pm'].astype(str)
df['day2_am'] = df['day2_am'].astype(str)
df['day2_pm'] = df['day2_pm'].astype(str)
#Discretize attribute ‘how many’ into bins of size 10.
max = df['how_many'].max()
min = df['how_many'].min()
print(min, max)
category = ['0-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80',
            '81-90','91-100','>100',]
df['How_many Category']=pd.cut(x=df['how_many'],
                      bins=[0,10,20,30,40,50,60,70,80,90,100,5000],
                      labels=category)
df = df.drop(columns=['how_many'])
#Randomly sample 100 instances without replacement
df_100_sample = df.sample(n=100, replace=False)
df_100_sample.to_csv('sampled_data.csv')