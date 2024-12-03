import pylightxl as xl
import pandas as pd
import os
import re
from dotenv import load_dotenv
load_dotenv()
def load_and_clean_data(directory : str) -> pd.DataFrame:
    clean_data = pd.DataFrame()
    total_sheets=[]
    sheet_count=0
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            db = xl.readxl(file_path)
            sheet_names = [i for i in db.ws_names if i != 'MAIN']
            total_sheets.extend(sheet_names)
            sheet_count+=len(sheet_names)
            for sheet in sheet_names:
                sheet_data = list(db.ws(ws=sheet).rows)
                df = pd.DataFrame(data=sheet_data[1:])
                data=df.iloc[0:6, 0:2]
                trans=data.T
                trans.columns=trans.iloc[0]
                trans=trans.drop(trans.index[0])
                trans = trans.rename(columns={trans.columns[0]: 'composition'})
                clean_data=pd.concat([clean_data,trans], ignore_index=True)  
    return clean_data, total_sheets
def parse_composition(composition):
    compounds = ['C2H6OSi', 'CdO', 'B4C', 'Gd2O3']
    quantities = {comp: 0 for comp in compounds}
    for comp in compounds:
        pattern = f'(\d+){comp}'
        match = re.search(pattern, composition)
        if match:
            quantities[comp] = int(match.group(1))
    return quantities




