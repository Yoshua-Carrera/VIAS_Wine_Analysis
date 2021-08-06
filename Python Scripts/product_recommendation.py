import pandas as pd
import numpy as np
from apyori import apriori
from matplotlib import pyplot as plt
from typing import List, Union, Dict
from pathlib import Path

class product_recommendation():
    def __init__(self, filename: str, *args):
        self.df = pd.read_csv(f'data/{filename}.csv')
        self.df_dict = {}
        for arg in args:
            self.df_dict[arg] = pd.read_csv(f'data/{arg}.csv', engine='python')
            
        
    
    def grouped_line_graph(self, data: pd.DataFrame, group_col: str, graph_col: str, xlabel: str, ylabel: str, title: str) -> pd.DataFrame:
        print('grouped_line_graph')
        document_count = data[[group_col, graph_col]].groupby(group_col).count().sort_values(by=graph_col, ascending=False).reset_index()
        plt.figure(figsize=(20, 10))
        plt.plot(document_count[graph_col])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.show()

        return document_count
    
    def subset_top_count(self, data: pd.DataFrame, subset_size: int, count_data: pd.DataFrame, count_label_col: str) -> pd.DataFrame:
        print('subset_top_count')
        document_subset = count_data.head(subset_size)[count_label_col]
        association_df = data[data[count_label_col].astype(str).isin(document_subset.tolist())].reset_index(drop=True)
        
        return association_df

    def dummy_and_group(self, data: pd.DataFrame, cols: List[str], group_col: str) -> pd.DataFrame:
        print('dummy_and_group')
        dummy_df = pd.get_dummies(data[cols]).reset_index(drop=True)
        updated_df = pd.concat([data, dummy_df], axis=1)
        updated_df = updated_df.groupby(group_col).sum()

        return updated_df

    def replace_dummmy_colname(self, data: pd.DataFrame) -> Union[pd.DataFrame, np.array]:
        print('replace_dummmy_colname')
        for col in data.columns:
            data[col] = data[col].astype(str).replace('1', col).replace('0', np.nan)

        association_list = data.to_numpy()

        association_list_clean = []

        for i in range(len(association_list)):
            association_list_clean.append(association_list[i][np.logical_not(pd.isnull(association_list[i]))])
        
        return data, association_list_clean

    def association_rules(self, data: np.array, min_sup: float, min_conf: float, min_lift: float, min_lenght: int) -> List:
        print('association_rules')
        association_rules = apriori(
            data, 
            min_support=min_sup, 
            min_confidence=min_conf, 
            min_lift=min_lift, 
            min_length=min_lenght
        )

        print('past bottleneck')

        association_results = list(association_rules)

        return association_results
    
    def display_association_rules(self, rule_list: List) -> Dict:
        print('display_association_rules')
        print('\nTotal of association rules found:', len(rule_list), '\n')
        results_dict = {}

        results_dict['rule'] = []
        results_dict['support'] = []
        results_dict['confidence'] = []
        results_dict['Lift'] = []
        
        for item in rule_list:
            pair_x = item[2][0][0]
            pair_y = item[2][0][1]
            items_x = [x for x in pair_x]
            items_y = [y for y in pair_y]
            
            results_dict['rule'].append( (str(items_x) + "->" + str(items_y)) ) 
            results_dict['support'].append( str(item[1]) ) 
            results_dict['confidence'].append( str(item[2][0][2]) ) 
            results_dict['Lift'].append( str(item[2][0][3]) ) 

            print("Rule: " + str(items_x) + "->" + str(items_y))
            
            print("Support: " + str(item[1]))
            print("Confidence: " + str(item[2][0][2]))
            print("Lift: " + str(item[2][0][3]))
            print("="*80, '\n')
    
        return results_dict

    def _write_rules(self, results_dict, filename: str) -> None:
        print('_write_rules')
        df = pd.DataFrame.from_dict(results_dict, orient='index').transpose()
        Path(r'Association results').mkdir(parents=True, exist_ok=True)
        df.to_csv(f'Association results/{filename}.csv')

    def execute_script(self, data: pd.DataFrame):
        print('execute_script')
        data = pd.merge(left=data, right=self.df_dict['item'][['No_', 'Description']], how='left', left_on='No_', right_on='No_')
        document_count = self.grouped_line_graph(data, 'Document No_', 'Description', 'Document Number', 'Line Count', 'Line Count By Invoice')
        association_df = self.subset_top_count(data, 500, document_count, 'Document No_')
        association_df = self.dummy_and_group(association_df, 'Description', 'Document No_')
        association_rules_df, association_rules_clean_list = self.replace_dummmy_colname(association_df)
        association_rules = self.association_rules(association_rules_clean_list, min_sup=0.1, min_conf=0.1, min_lift=3, min_lenght=2)
        association_rules_dict = self.display_association_rules(association_rules)
        self._write_rules(association_rules_dict, 'association_rules_df')

if __name__=='__main__':
    PR = product_recommendation('Sales_Invoice_Line', 'item')
    PR.execute_script(PR.df)