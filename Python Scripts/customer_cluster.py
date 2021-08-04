import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

import seaborn

from scipy.cluster.hierarchy import dendrogram, linkage

import plotly.figure_factory as ff
import plotly.offline as pltoff
import plotly as py

import numpy as np

from typing import List

from pathlib import Path

import pickle as pk

class data_cluster:
    def __init__(self):
        self.customer_df = pd.read_csv(r'Clean_Data/Customer_clean.csv')
        self.sales_invoice_header_df = pd.read_csv(r'Clean_Data/Sales_Invoice_Header_Clean.csv')
        self.credit_invoice_header_df = pd.read_csv(r'Clean_Data/Sales_Cr_Invoice_Header_Clean.csv')
        self.value_entry = pd.read_csv(r'data/ValueEntry.csv')
        self.item_ledger_entry = pd.read_csv(r'data/ItemLedgerEntry.csv')
        self.customer_df_no_dup: pd.DataFrame
        self.customer_joined: pd.DataFrame
    
    def fix_duplicates(self, table: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
        df = table
        for col in col_list:
            # Mark Duplicated Data
            df[f'{col}Dup'] = df[col].duplicated() # No_ duplicate booleans
            # Take Out duplicated keys
            duplicate_list = df[df[f'{col}Dup']][col].astype(str).tolist() # List of duplicate values
            df = df[~df[col].isin(duplicate_list)] # Filtering
        
        return df
        

    def join_summarize_data(self, group_df: pd.DataFrame, group_cols: List[str], group_var: str, join_df: pd.DataFrame, left_on: str, right_on:str, how='left'):
        # Summarize And Join Data
        df_summmarized = group_df[group_cols].groupby(group_var).sum()

        df_joined = join_df.merge(df_summmarized, 
                                    left_on=left_on, 
                                    right_on=right_on, 
                                    how=how)

        return df_joined

    def create_dummies(self, df: pd.DataFrame, col: str, col_list: List[str]) -> pd.DataFrame:
        dendrogram_data = pd.get_dummies(df[col])
        dendrogram_data = pd.concat([df, dendrogram_data], axis=1)
        dendrogram_data = dendrogram_data[col_list]
        
        return dendrogram_data

    def create_dendrogram(self, data: pd.DataFrame, pkg: str='scipy', subset: bool=False, subset_n: int=500, 
                            plot:bool=True, drop_na: bool=True, log_transform: List[str]=None, standard_scale: List[str]=None):
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
        if drop_na:
            # Ignore Customers without data
            dendrogram_data = data.dropna(how='any').reset_index(drop=True)
        # Log transform sales
        if log_transform:
            for col in log_transform:
                dendrogram_data[col] = np.log(dendrogram_data[col])
        
        if standard_scale:
            new_cols = {}

            if standard_scale[0] == '*':
                for col in dendrogram_data.columns:
                    new_cols[col] = f'{col}_scaled'  

                scaled_cols = StandardScaler().fit_transform(dendrogram_data)
                dendrogram_data_scaled = pd.DataFrame(data=scaled_cols, columns=dendrogram_data.columns).rename(columns=new_cols)
                # dendrogram_data_scaled = pd.concat([dendrogram_data, dendrogram_data_scaled], axis=1)
            else:    
                for col in standard_scale:
                    new_cols[col] = f'{col}_scaled'  

                scaled_cols = StandardScaler().fit_transform(dendrogram_data[standard_scale])
                dendrogram_data_scaled = pd.DataFrame(scaled_cols, columns=standard_scale)
                dendrogram_data_scaled = pd.concat([dendrogram_data.drop(standard_scale, axis=1), dendrogram_data_scaled], axis=1).rename(columns=new_cols)
        
        # plot
        if plot:
            if pkg == 'scipy':
                if subset:
                    Z = linkage(dendrogram_data.head(subset_n), 'ward')
                    fig = plt.figure(figsize=(25, 10))
                    dn = dendrogram(Z)
                    print('dn', dn)
                    plt.show()
                else:
                    Z = linkage(dendrogram_data, 'ward')
                    fig = plt.figure(figsize=(25, 10))
                    dn = dendrogram(Z)
                    print('dn', dn)
                    plt.show()

            else:
                if subset:
                    fig = ff.create_dendrogram(dendrogram_data.head(subset_n))
                    fig.update_layout(width=2000, height=500)
                    py.offline.plot(fig)
                else:
                    fig = ff.create_dendrogram(dendrogram_data)
                    fig.update_layout(width=2000, height=500)
                    py.offline.plot(fig)
        print(dendrogram_data_scaled.info)
        return dendrogram_data_scaled

    def compute_df_stats(self, df: pd.DataFrame, col_list: List[str]):
        for var in col_list:
            print('{} {}:  {}'.format(var, 'customer_dendrogram.head(500)', df.head(500)[var].mean()))
            print('{} {}:  {}'.format(var, 'customer_dendrogram.tail(500)', df.tail(500)[var].mean()))
            print('{} {}:  {}'.format(var, 'customer_dendrogram.head(1500)', df.head(1500)[var].mean()))
            print('{} {}:  {}'.format(var, 'customer_dendrogram.tail(1500)', df.tail(1500)[var].mean()))
            print('{} {}:  {}'.format(var, 'customer_dendrogram', df[var].mean()))

    def plot_cluster_elbow(self, data: pd.DataFrame, n_clusters: int):
        sum_osd = []

        K = range(1,30)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(data)
            sum_osd.append(km.inertia_) # Inertia method includes the stat

        plt.plot(K, sum_osd, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def compute_km_cluster(self, data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
        km = KMeans(n_clusters=n_clusters).fit(data)
        data['cluster_group'] = km.labels_ 
        return data
    
    def write_cluster_data(self, data: List[pd.DataFrame], filename: str) -> None:
        if len(data) > 1:
            write_data = pd.concat(data, axis=1)
            Path(r'Clustered Data').mkdir(parents=True, exist_ok=True)
            print(write_data.info)
            write_data.to_csv(f'Clustered Data/{filename}.csv')
        else:
            Path(r'Clustered Data').mkdir(parents=True, exist_ok=True)
            print(data[0].info)
            data[0].to_csv(f'Clustered Data/{filename}.csv')

    def pca(self, data: pd.DataFrame, n_comp: int, columns: List[str]=None) -> pd.DataFrame:
        pca = PCA(n_components=n_comp)
        data = data.reset_index(drop=True)
        if columns is None:
            pca_fit = pca.fit_transform(data.dropna())
        else:
            pca_fit = pca.fit_transform(data[columns].dropna())

        pca_dataset = pd.DataFrame(data = pca_fit, columns=['component_1', 'component_2'])

        pca_dataset = pd.concat([pca_dataset, data], axis=1)

        Path(r'Models').mkdir(parents=True, exist_ok=True)
        
        if ~Path('Models/pca.pkl').is_file():
            pk.dump(pca_fit, open(f'Models/pca.pkl',"wb"))

        return pca_dataset

    def cluster_scatter(self, data: pd.DataFrame, components: List[str]) -> None:
        
        clusters = data['cluster_group'].unique()
        fg = seaborn.FacetGrid(data=data, hue='cluster_group', hue_order=clusters, aspect=1.61)
        fg.map(plt.scatter, components[0], components[1]).add_legend()

    def exectute_script(self):
        no_dup_customer_df =  self.fix_duplicates(self.customer_df, ['No_', 'Name'])
        item_ledger_entry_joined = self.join_summarize_data(
            self.value_entry,
            ['Item Ledger Entry No_', 'Sales Amount (Actual)'],
            'Item Ledger Entry No_',
            self.item_ledger_entry,
            'Entry No_',
            'Item Ledger Entry No_'
        )
        customer_joined_df = self.join_summarize_data(
            item_ledger_entry_joined,
            ['Source No_', 'Sales Amount (Actual)'],
            'Source No_',
            no_dup_customer_df,
            'No_',
            'Source No_'
        )
        dendrogram_data = self.create_dummies(
            customer_joined_df, 
            'On premise/off premise',
             ['latitude', 'longitude', 'Sales Amount (Actual)', 'Distributors', 'Off premise', 'On premise', 'Others']
        )

        pca_data = self.pca(dendrogram_data.dropna(), 2)
        
        dendrogram_plot_data = self.create_dendrogram(pca_data[['component_1', 'component_2']], plot=False, subset=False, standard_scale=['*'])
        
        self.compute_df_stats(
            dendrogram_data,
            ['latitude', 'longitude', 'Sales Amount (Actual)', 'Distributors', 'Off premise', 'On premise', 'Others']
        )

        self.plot_cluster_elbow(dendrogram_plot_data, 30)

        clustered_data = self.compute_km_cluster(dendrogram_plot_data, 5)

        self.cluster_scatter(clustered_data, ['component_1_scaled', 'component_2_scaled'])
        
        self.write_cluster_data([clustered_data, pca_data], 'clustered_data')

if __name__=='__main__':
    customer_cluster = data_cluster()
    customer_cluster.exectute_script()