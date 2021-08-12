import pandas as pd
import numpy as np
from apyori import apriori
from matplotlib import pyplot as plt
from typing import List, Union, Dict
from pathlib import Path

class product_recommendation():
    '''
    Product recommendation Class: object class that takes a pandas dataframe containing product order data in order to compute association rules

    Mehotds
        - grouped_line_graph
            * Takes a pandas DF, list of colmns to group by, column name to graph, graph label strings
            * The df input structure contains 1 row per order detail, so it is grouped by invoice number in order to count the number of lines
            per invoice code
            * Renders a line plot showing the line count per document
            * Return a summarized dataset that shows the line count per document

        - subset_top_count
            * The purpose of this method is to subset the invoices and select only the top (by line count) of them
            * Takes a complete DF to be subset, a DF containing the invoice and corresponding line count, a subset size integer
            a label for the count colum
            * Subsets the cont DF, and extracts the document numbers in order to filter the complete dataframe
            * Returns the filtered "complete" DF

        - dummy_and_group
            * The purpose of this method is to "dummify" the column containing the product code and then group the DF by invoice number in order
            to have a DF in the following structure:

                                    Invoice	 Product 1	Product 2	Product …	Product n
                                        1	|    1	  |       0	  |           |     1     |
                                        2	|    0	  |       0	  |           |     1     |
                                        …	|    	  |       	  |           |           |
                                        n	|    1	  |       1	  |           |     1     |

            here "0" means the invoice does not contain the product and "1" means the invoice contains the number.

            * A DF is returned in the structure shown above

        - replace_dummmy_colname
            * Parts from the output shown in the previous method (only input taken), then replaces every "1" with the column name and every "0" 
            eith a NaN value (missing value)
            * A DF is returned in the structure shown above, and the same data in list form
        
        - association_rules
            * This method trains the model itself, it takes data in array form where each individual array contains as many items as there are
            products in an order. It looks like this:
                        [
                            [Product A, Product F, Product E, ..., Product ?],
                            [Product B, Product I, Product P, ..., Product ?],
                        ]
            Arrays have different lengths, and it depends on the number of lines each invoice has
            
            * This method also take numrical values for min_sup, min_conf, min_lift, min_length
            * it is convenient to do a quick summary of the main stadistics computed:
                ~ We define 2 item sets X and Y where thre exists an association rule given certain parameter thresholds, X and Y
                dont necesarilly have the same length. (X => Y)
                ~ Support is calculated by measuring the proportion of the orders that contain both X and Y
                ~ Confidence is the numbers of orders that contain both X and Y divided by the the orders that contain X
                ~ Lift is confidence/support, an it is the likelihood of getting X and Y together rather than just Y
                ~ Length is the number of elements in a rule
                ~ E.G: {Wine, Beer} => {Scotch} ---- Support=0.4, confidence=0.67, Lift=1.675, length=3
                A client is 1.675 more likely to get wine, beer and scoth, rather than just scotch
            * The method returns a nested list containing all the rules generated at the given thresholds
        
        - display_association_rules
            * This method takes the association rules nested list and displays the each rule, with its parameters
            * It also saves all the important information in a dictionary that is finally returned

        - _write_rules
            * This method writes the association rules in a CSV file by using the association rules dictionary returned
            by the previous method

        - execute_script
            * Executes all the methods of the class
            * Here the itemcode is replaced by the item description in order to overlook little product variations and have
            more practical association rules
    '''
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

    def execute_script(self, data: pd.DataFrame) -> pd.DataFrame:
        print('execute_script')
        data = pd.merge(left=data, right=self.df_dict['item'][['No_', 'Description']], how='left', left_on='No_', right_on='No_')
        document_count = self.grouped_line_graph(data, 'Document No_', 'Description', 'Document Number', 'Line Count', 'Line Count By Invoice')
        association_df = self.subset_top_count(data, 500, document_count, 'Document No_')
        association_df = self.dummy_and_group(association_df, 'Description', 'Document No_')
        association_rules_df, association_rules_clean_list = self.replace_dummmy_colname(association_df)
        association_rules = self.association_rules(association_rules_clean_list, min_sup=0.1, min_conf=0.1, min_lift=3, min_lenght=2)
        association_rules_dict = self.display_association_rules(association_rules)
        self._write_rules(association_rules_dict, 'association_rules_df')
        return association_rules_df

if __name__=='__main__':
    PR = product_recommendation('Sales_Invoice_Line', 'item')
    PR.execute_script(PR.df)