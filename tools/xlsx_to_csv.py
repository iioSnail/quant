import os

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    files = os.listdir("../data")
    for file in tqdm(files):
        if not file.endswith(".xlsx"):
            continue

        df = pd.read_excel(f"../data/{file}", dtype=str)

        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        df.to_csv(f"../data/{file.replace('.xlsx', '.csv')}", index=False)