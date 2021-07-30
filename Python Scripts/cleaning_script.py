import pandas as pd
import pgeocode
import re
from pyzipcode import ZipCodeDatabase

from Metadata.us_state_abbrev import us_state_abbrev

# Customer table cleaning

Customerdf = pd.read_csv(r'../data/Customer.csv')
nomi = pgeocode.Nominatim('US')

def parse_to_coordinates(row):
    try:
        coordinate = nomi.query_postal_code(row['Post Code'])
        latitude = coordinate.latitude
        longitude = coordinate.longitude
        print(longitude, latitude, [longitude, latitude])
        return longitude, latitude, [longitude, latitude]
    except:
        return None, None, None

Customerdf['longitude'], Customerdf['latitude'], Customerdf['coordinates'] = zip(*Customerdf.apply(lambda row: parse_to_coordinates(row), axis=1))

print(Customerdf[['longitude', 'latitude', 'coordinates']])

Customerdf.to_csv(r'Clean_Data/Customer.csv')

# Sales/Credit invoice header cleaning 

sales_invoice_header_df = pd.read_csv(r'../data/Sales_Invoice_Header.csv')
credit_invoice_header_df = pd.read_csv(r'../data/Sales_Cr_Invoice_Header.csv')

def clean_ship_to_county(row):
    try:
        Ship_to_County = row['Ship-to County'].strip()
    except:
        return row['Ship-to County']

    if Ship_to_County.upper() in us_state_abbrev.values():
        return row['Ship-to County']

    elif Ship_to_County.capitalize() in us_state_abbrev.keys():
        return us_state_abbrev[Ship_to_County.capitalize()]
    
    else:
        zip_pattern = re.compile(r'/d{5}')
        if zip_pattern.search(Ship_to_County):
            zcdb = ZipCodeDatabase()
            zipcode = zcdb[int(Ship_to_County)]
            return zipcode
        return Ship_to_County

sales_invoice_header_df['clean Ship-to County'] = sales_invoice_header_df.apply(lambda row: clean_ship_to_county(row), axis=1) 
credit_invoice_header_df['clean Ship-to County'] = credit_invoice_header_df.apply(lambda row: clean_ship_to_county(row), axis=1)

sales_invoice_header_df.to_csv(r'Clean_Data/Sales Invoice Header.csv')
credit_invoice_header_df.to_csv(r'Clean_Data/Credit Invoice Header.csv')