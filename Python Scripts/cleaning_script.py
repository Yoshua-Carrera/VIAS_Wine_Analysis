import pandas as pd
import pgeocode
import re
from pyzipcode import ZipCodeDatabase
from pathlib import Path
from typing import Union, Tuple, List, Iterable

from Metadata.us_state_abbrev import us_state_abbrev

class cleaning_script():
    def __init__(self, filename: str):
        self.df = pd.read_csv(f'data/{filename}.csv')
        self.nomi = pgeocode.Nominatim('US')

    def coordinates(self, colname) -> None:
        self.df['longitude'], self.df['latitude'], self.df['coordinates'] = zip(*self.df.apply(lambda row: self.__parse_coordinates(row, colname), axis=1))

    def __parse_coordinates(self, row, colname) -> Iterable[Union[None, int, List]]:
        try:
            coordinate = self.nomi.query_postal_code(row[colname])
            latitude = coordinate.latitude
            longitude = coordinate.longitude
            return longitude, latitude, [longitude, latitude]
        except:
            return None, None, None
    
    def write_csv(self, filename: str) -> None:
        Path(r'Clean_Data').mkdir(parents=True, exist_ok=True)
        self.df.to_csv(f'Clean_Data/{filename}.csv')

    def __parse_state(self, row, colname: str) -> str:
        try:
            state_var = row[colname].strip()
        except:
            return row[colname]

        if state_var.upper() in us_state_abbrev.values():
            return row[colname]

        elif state_var.capitalize() in us_state_abbrev.keys():
            return us_state_abbrev[state_var.capitalize()]
        
        else:
            zip_pattern = re.compile(r'/d{5}')
            if zip_pattern.search(state_var):
                zcdb = ZipCodeDatabase()
                zipcode = zcdb[int(state_var)]
                return zipcode
        return state_var

    def state(self, colname: str) -> None:
        self.df[f'Clean {colname}'] = self.df.apply(lambda row: self.__parse_state(row, colname), axis=1) 

cleaner = cleaning_script('Customer')
cleaner.coordinates('Post Code')
cleaner.write_csv('Customer_clean')

cleaner = cleaning_script('Sales_Invoice_Header')
cleaner.state('Ship-to County')
cleaner.write_csv('Sales_Invoice_Header_Clean')

cleaner = cleaning_script('Sales_Cr_Invoice_Header')
cleaner.state('Ship-to County')
cleaner.write_csv('Sales_Cr_Invoice_Header_Clean')