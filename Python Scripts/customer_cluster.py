import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

import plotly.figure_factory as ff
import plotly.offline as pltoff
import plotly as py

import numpy as np


# Import Data 
customer_df = pd.read_csv(r'Clean_Data/Customer.csv')
sales_invoice_header_df = pd.read_csv(r'Clean_Data/Sales Invoice Header.csv')
credit_invoice_header_df = pd.read_csv(r'Clean_Data/Credit Invoice Header.csv')
value_entry = pd.read_csv(r'../data/ValueEntry.csv')
item_ledger_entry = pd.read_csv(r'../data/ItemLedgerEntry.csv')

# Mark Duplicated Data
customer_df['No_Dup'] = customer_df['No_'].duplicated()
customer_df['NameDup'] = customer_df['Name'].duplicated()

# Take Out duplicated keys
duplicate_No = customer_df[customer_df['No_Dup']]['No_'].astype(str).tolist()
customer_df_no_dup = customer_df[~customer_df['No_'].isin(duplicate_No)]

# Summarize And Join Data
value_entry_summmarized = value_entry[['Item Ledger Entry No_', 'Sales Amount (Actual)']].groupby('Item Ledger Entry No_').sum()

item_ledger_entry_joined = item_ledger_entry.merge(value_entry_summmarized, 
                                                    left_on='Entry No_', 
                                                    right_on='Item Ledger Entry No_', 
                                                    how='left')

item_ledger_entry_summarized = item_ledger_entry_joined[['Source No_', 'Sales Amount (Actual)']].groupby('Source No_').sum()

customer_joined = customer_df.merge(item_ledger_entry_summarized, 
                                    left_on='No_', 
                                    right_on='Source No_',
                                     how='left')

# Dendrogram
'''
['Unnamed: 0', 'No_', 'Name', 'Address', 'City',
       'Customer Posting Group', 'Salesperson Code', 'Shipment Method Code',
       'Country_Region Code', 'Gen_ Bus_ Posting Group', 'County', 'CityState',
       'Customer', 'StateCountry', 'Post Code', 'GeoColumn With Address',
       'GeoColumn', 'State Area Code', 'State Zone Code',
       'State Location Code', 'Mkt Segment', 'On premise/off premise',
       'longitude', 'latitude', 'coordinates', 'No_Dup', 'NameDup',
       'Sales Amount (Actual)']
'''
## Dummy creation
customer_dendrogram = pd.get_dummies(customer_joined['On premise/off premise'])
customer_dendrogram = pd.concat([customer_joined, customer_dendrogram], axis=1)
customer_dendrogram = customer_dendrogram[['latitude', 'longitude', 'Sales Amount (Actual)', 
                                        'Distributors', 'Off premise', 'On premise', 'Others']]
## Ignore Customers without data
# customer_dendrogram = customer_dendrogram[~customer_dendrogram['Sales Amount (Actual)'].isna()]
customer_dendrogram = customer_dendrogram.dropna(how='any')
customer_dendrogram = customer_dendrogram[customer_dendrogram['Sales Amount (Actual)']>0]
customer_dendrogram['Sales Amount (Actual)'] = np.log(customer_dendrogram['Sales Amount (Actual)'])

## Min Max scaling

# mms = MinMaxScaler()
# mms.fit(customer_dendrogram)
# customer_dendrogram = mms.transform(customer_dendrogram.head(500))

fig = ff.create_dendrogram(customer_dendrogram.tails(500))
fig.update_layout(width=2000, height=500)
py.offline.plot(fig)

## Scipy clustering

Z = linkage(customer_dendrogram.head(500), 'ward')

for var in ['latitude', 'longitude', 'Sales Amount (Actual)', 'Distributors', 'Off premise', 'On premise', 'Others']:
       print('{} {}:  {}'.format(var, 'customer_dendrogram.head(500)', customer_dendrogram.head(500)[var].mean()))
       print('{} {}:  {}'.format(var, 'customer_dendrogram.tail(500)', customer_dendrogram.tail(500)[var].mean()))
       print('{} {}:  {}'.format(var, 'customer_dendrogram.head(1500)', customer_dendrogram.head(1500)[var].mean()))
       print('{} {}:  {}'.format(var, 'customer_dendrogram.tail(1500)', customer_dendrogram.tail(1500)[var].mean()))
       print('{} {}:  {}'.format(var, 'customer_dendrogram', customer_dendrogram[var].mean()))

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.show()  

# Scikit clustering
## Scikit K-Means

sum_osd = [] # sum of squared distances

# K = range(1,30)
# for k in K:
#        km = KMeans(n_clusters=k)
#        km = km.fit(customer_dendrogram)
#        sum_osd.append(km.inertia_) # Inertia method includes the stat

# plt.plot(K, sum_osd, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Sum_of_squared_distances')
# plt.title('Elbow Method For Optimal k')
# plt.show()
