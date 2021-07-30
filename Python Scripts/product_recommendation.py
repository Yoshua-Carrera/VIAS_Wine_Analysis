import pandas as pd
import numpy as np
from apyori import apriori
from matplotlib import pyplot as plt

class product_recommendation():
    def __init__(self, filename: str):
        self.df = pd.read_csv(f'../data/{filename}.csv')