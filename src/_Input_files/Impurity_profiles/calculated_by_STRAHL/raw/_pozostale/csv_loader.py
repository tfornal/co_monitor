# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:26:33 2021

@author: t_fornal
"""

import pandas as pd
import glob
import natsort 

V = 0
file_name = [
    # '20181009.034_1680ms', ###peaked C profile
    # '20180816_022@3_9500', ### 8000eV, 2e13 
    '20181011_012@5_5000' ### 1750 eV, 7e13
    ]
ion = "C6+"



path = fr'D:\1. PRACA\1. IFPiLM\_Doktorat\____Analizy\Analiza_2\STRAHL_results\{file_name[0]}\V={V}' # use your path
print(path)
# path = r'D:\1. PRACA\1. IFPiLM\_Doktorat\____Analizy\Analiza_2\STRAHL_results\20181009.034_1680ms\V=-100' # use your path
all_files = glob.glob(path + "/*.csv")
all_files = natsort.natsorted(all_files,reverse=False)
names = []
li = []

for filename in all_files:
    df = pd.read_csv(filename, usecols=[ion], sep = ";")
    li.append(df)
    names.append(filename)
    


names = natsort.natsorted(names,reverse=False)

all_files = all_files[:] 

df = pd.concat(li, axis=1, ignore_index=True)
for filename in all_files:
    xx = pd.read_csv(filename, usecols=["r/a"], sep = ";")
df['r/a'] = xx
col_name="r/a"
first_col = df.pop("r/a")
df.insert(0, col_name, first_col)
import matplotlib.pyplot as plt

for i in range(len(all_files)):
    # plt.close('all')
    plt.plot(df["r/a"], df[i])
    # plt.show()






nnn = df.to_numpy()
nnn= nnn[:,1:]

def d3_plot():
        
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # Grab some test data.
    X = df.to_numpy()[:,0]
    Y = np.arange(24)
    Z = nnn[:,1:]
    x, y = np.meshgrid(X,Y)
    x = x.T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x,y.T,nnn)
    plt.show()

d3_plot()









