# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:26:33 2021

@author: t_fornal
"""

import pandas as pd
import glob
import natsort 
import os

V = 0
file_name = [
    # '20181009.034_1680', ###peaked C profile
    # '20180816_022@3_9500', ### 8000eV, 2e13 
    '20181011_012@5_5000' ### 1750 eV, 7e13
    ]
ion = "C_total"

path = fr'D:\1. PRACA\_Programy\CO_Monitor\COMonitor_simulation\_Input_files\Impurity_profiles\calculated_by_STRAHL\raw\{file_name[0]}\V={V}'
    
all_files = glob.glob(path + "/*.csv")
all_files = natsort.natsorted(all_files,reverse=False)
column_names = []
profiles_list = []

for filename in all_files:
    df = pd.read_csv(filename, usecols=[ion], sep = ";")
    profiles_list.append(df)
    column_names.append(os.path.basename(filename)[-11:-6])
new_column_names = [s.replace("-", "") for s in column_names]

df = pd.concat(profiles_list, axis=1, ignore_index=True)
df.columns = new_column_names

df['r/a'] = pd.read_csv(all_files[0], usecols=["r/a"], sep = ";")

def move_last_column_to_front():
    col_name="r/a"
    first_col = df.pop("r/a")
    df.insert(0, col_name, first_col)
    
    return df
df = move_last_column_to_front()

df.to_csv(f"{file_name[0]}_V={V}_all.csv", sep = ";")
df = df.drop((df[df["r/a"] > 1]).index)
df.to_csv(f"{file_name[0]}_V={V}.csv", sep = ";")

