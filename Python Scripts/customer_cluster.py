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

from os import path

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

    def create_dummies(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        dendrogram_data = pd.get_dummies(df[col])
        dendrogram_data = pd.concat([df, dendrogram_data], axis=1)
        dendrogram_data = dendrogram_data.reset_index(drop=True)
        
        return dendrogram_data

    def create_dendrogram(self, data: pd.DataFrame, pkg: str='scipy', plot_cols: List[str]=None, subset: bool=False, subset_n: int=500, 
                            plot:bool=True, drop_na: bool=True, log_transform: List[str]=None, standard_scale: List[str]=None, uid: str=None,
                            cluster_cols: List[str]=None):
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
        if plot_cols is None:
            plot_cols = list(data.columns)

        dendrogram_data = data[cluster_cols]

        if drop_na:
            # Ignore Customers without data
        
        # if uid:
            data = data.dropna(subset=cluster_cols).reset_index(drop=True).drop(columns=[uid])
            dendrogram_data = dendrogram_data.dropna(subset=cluster_cols).reset_index(drop=True)
        # else:
            # dendrogram_data = data.dropna(subset=cluster_cols).reset_index(drop=True)
        
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
                dendrogram_data = pd.DataFrame(data=scaled_cols, columns=dendrogram_data.columns).rename(columns=new_cols)
            else:    
                for col in standard_scale:
                    new_cols[col] = f'{col}_scaled'  

                scaled_cols = StandardScaler().fit_transform(dendrogram_data[standard_scale])
                dendrogram_data = pd.DataFrame(scaled_cols, columns=standard_scale)
                dendrogram_data = pd.concat([dendrogram_data.drop(standard_scale, axis=1), dendrogram_data], axis=1).rename(columns=new_cols)
        
        # plot
        if plot:
            if pkg == 'scipy':
                if subset:
                    Z = linkage(dendrogram_data.head(subset_n)[plot_cols], 'ward')
                    fig = plt.figure(figsize=(25, 10))
                    dn = dendrogram(Z)
                    print('dn', dn)
                    plt.show()
                else:
                    Z = linkage(dendrogram_data[plot_cols], 'ward')
                    fig = plt.figure(figsize=(25, 10))
                    dn = dendrogram(Z)
                    print('dn', dn)
                    plt.show()

            else:
                if subset:
                    fig = ff.create_dendrogram(dendrogram_data.head(subset_n)[plot_cols])
                    fig.update_layout(width=2000, height=500)
                    py.offline.plot(fig)
                else:
                    fig = ff.create_dendrogram(dendrogram_data[plot_cols])
                    fig.update_layout(width=2000, height=500)
                    py.offline.plot(fig)
        print(dendrogram_data.info)

        if uid in data.columns:
            data = data.dropna(how='any').reset_index(drop=True)
            dendrogram_data_complete = pd.concat([data, dendrogram_data], axis=1)

            return dendrogram_data, dendrogram_data_complete
        
        dendrogram_data_complete = pd.concat([dendrogram_data, data], axis=1)
        
        return dendrogram_data, dendrogram_data_complete

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

    def pca(self, data: pd.DataFrame, n_comp: int, uid: str, modelname: str, columns: List[str]=None) -> pd.DataFrame:
        pca = PCA(n_components=n_comp)
        data = data.reset_index(drop=True)
        uid_data = data[uid]
        if columns is None:
            data = data.dropna().drop(uid, axis=1)
            pca_fit = pca.fit_transform(data)
        else:
            list_and_uid = columns + [uid]
            data = data[list_and_uid].dropna().reset_index(drop=True)
            uid_data = data[uid]
            pca_fit = pca.fit_transform(data[columns])

        pca_dataset = pd.DataFrame(data = pca_fit, columns=['component_1', 'component_2'])

        pca_dataset = pd.concat([pca_dataset, data, uid_data], axis=1)

        Path(r'Models').mkdir(parents=True, exist_ok=True)
        
        if not path.exists(f'Models/{modelname}.pkl'):
            pk.dump(pca_fit, open(f'Models/{modelname}.pkl',"wb"))

        return pca_dataset

    def cluster_scatter(self, data: pd.DataFrame, components: List[str], xlabel: str=None, ylabel: str=None) -> None:
        
        clusters = data['cluster_group'].unique()
        fg = seaborn.FacetGrid(data=data, hue='cluster_group', hue_order=clusters, aspect=1.61)
        fg.map(plt.scatter, components[0], components[1]).add_legend()
        fg.set(yticks=[], xticks=[])
        plt.show()

    def exectute_script(self, no_dup_cols: List[str], dummy_col: List[str], dendrogram: bool, scatter: bool, scatter_cols: List[str], uid: str, filename: str, 
                        pca: bool=True, cluster_cols: List[str]=None, standard_scale: List[str]=None, scatter_axis_labs: List[str]=None):
        no_dup_customer_df =  self.fix_duplicates(self.customer_df, no_dup_cols)
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
            customer_joined_df[~customer_joined_df['On premise/off premise'].isin(['Distributors', 'Others'])],
            dummy_col
        )

        if pca:
            pca_data = self.pca(dendrogram_data, 
                                2, 
                                uid='No_', 
                                columns=cluster_cols,
                                modelname='pca')
            
            dendrogram_plot_data = self.create_dendrogram(pca_data[['component_1', 'component_2']], 
                                                            plot=dendrogram, subset=False, standard_scale=['*'])
            
            self.compute_df_stats(
                pca_data,
                cluster_cols
            )

            self.plot_cluster_elbow(dendrogram_plot_data, 30)

            clustered_data = self.compute_km_cluster(dendrogram_plot_data, 5)

            self.write_cluster_data([clustered_data, pca_data], 'clustered_data')

            if scatter:
                if scatter_axis_labs: #['component_1_scaled', 'component_2_scaled'] ------ ['Component 1 Scaled', 'Component 2 Scaled']
                    xlab, ylab = scatter_axis_labs[0], scatter_axis_labs[1]
                    self.cluster_scatter(clustered_data, scatter_cols, xlab, ylab)
                else:
                    self.cluster_scatter(clustered_data, scatter_cols)

        else:
            dendrogram_plot_data, dendrogram_plot_data_complete = self.create_dendrogram(dendrogram_data,
                                                                                        cluster_cols=cluster_cols,
                                                                                        plot=dendrogram, 
                                                                                        subset=False, 
                                                                                        standard_scale=standard_scale,
                                                                                        uid=uid)
            
            self.compute_df_stats(
                dendrogram_data,
                cluster_cols
            )

            self.plot_cluster_elbow(dendrogram_plot_data, 30)

            clustered_data = self.compute_km_cluster(dendrogram_plot_data_complete[cluster_cols], 5)

            write_data = pd.concat([clustered_data['cluster_group'], dendrogram_plot_data_complete], axis=1)

            self.write_cluster_data([write_data], filename) # str(datetime.now().strftime('%Y%m%d%H%M%S'))

            if scatter:
                if scatter_axis_labs:
                    xlab, ylab = scatter_axis_labs[0], scatter_axis_labs[1]
                    self.cluster_scatter(write_data, scatter_cols, xlab, ylab)
                else:
                    self.cluster_scatter(write_data, scatter_cols)
            

if __name__=='__main__':
    customer_cluster = data_cluster()
    customer_cluster.exectute_script(no_dup_cols=['No_', 'Name'], 
                                    dummy_col='On premise/off premise',
                                    cluster_cols=['latitude', 'longitude', 'Sales Amount (Actual)', 'Off premise', 'On premise'],
                                    dendrogram=False,
                                    scatter= False,
                                    scatter_cols=['Sales Amount (Actual)_scaled', 'No_'],
                                    scatter_axis_labs=['Sales', 'Client'],
                                    filename='clustered_data',
                                    uid='No_',
                                    standard_scale=['*'],
                                    pca=False)
    print('*'*50)
    print('*'*50)
    print('*'*50)
    customer_cluster.exectute_script(no_dup_cols=['No_', 'Name'], 
                                    dummy_col='On premise/off premise',
                                    cluster_cols=['Sales Amount (Actual)'],
                                    dendrogram=False,
                                    scatter= False,
                                    scatter_cols=['Sales Amount (Actual)_scaled', 'No_'],
                                    scatter_axis_labs=['Sales', 'Client'],
                                    filename='clustered_data_sales_only',
                                    uid='No_',
                                    standard_scale=['*'],
                                    pca=False)
    