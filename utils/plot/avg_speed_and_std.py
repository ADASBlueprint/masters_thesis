import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

#graphing options
#1 = cumulative collision rate during train
option = 1

#read data
VERSION2 = sys.argv[2]
VERSION3 = sys.argv[3]

v2_filename = "data-" + VERSION2 + ".csv"
v3_filename = "data-" + VERSION3 + ".csv"

cwd = os.path.dirname(os.path.realpath(__file__))
parent = Path(cwd)
# assume root directory is two folders up
root = parent.parent.absolute().parent.absolute()
 
v2_data_file = str(root) +"\\dqn_learning_highway_v2\\data\\"+ v2_filename
v3_data_file = str(root) +"\\dqn_learning_highway_v3\\data\\"+ v3_filename

data_df = pd.read_csv(v2_data_file)
data2_df = pd.read_csv(v3_data_file)

smooth = 10

if sys.argv[1] == "spd":
    #remove collison rows
    # for idx in data_df.index.values:
    #     if data_df['coll'].loc[idx] == 1 or data2_df['coll'].loc[idx] == 1:
    #         data_df.drop([idx], inplace = True)
    #         data2_df.drop([idx], inplace = True)
    data_df.reset_index(inplace = True)
    data2_df.reset_index(inplace = True)
    data_df['avg_spd']=data_df['avg_spd'].rolling(smooth).mean()*1.42
    data2_df['avg_spd']=data2_df['avg_spd'].rolling(smooth).mean()*1.42
    data_df['spd_std_dev']=data_df['spd_std_dev'].rolling(smooth).mean()
    data2_df['spd_std_dev']=data2_df['spd_std_dev'].rolling(smooth).mean()
    #plot data
    fig, axes = plt.subplots(nrows=2,  ncols=1)
    data_df['avg_spd'].dropna().plot(kind='line',y=1, use_index=True, color='purple',legend = True, ax=axes[0])
    data2_df['avg_spd'].dropna().plot(kind='line',y=1, use_index=True, color='orange', legend = True, ax=axes[0])
    # ax2=ax.twinx()
    data_df['spd_std_dev'].dropna().plot(kind='line',y=1, use_index=True, color='purple',legend = True, ax=axes[1])
    data2_df['spd_std_dev'].dropna().plot(kind='line',y=1, use_index=True, color='orange',legend = True, ax=axes[1])

    axes[0].title.set_text("Average Speed m/s")
    axes[0].set_ylabel("avg speed")
    axes[1].title.set_text("Standard Deviation of Speed")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("std deviation")
    axes[0].legend(["w/o 2nd Perception Vehicles","w/ 2nd Perception Vehicles"], loc='lower right')
    axes[1].legend(["w/o 2nd Perception Vehicles","w/ 2nd Perception Vehicles"], loc='upper right')
else:
    # #remove collison rows
    # for idx in data_df.index.values:
    #     if data_df['coll'].loc[idx] == 1 or data2_df['coll'].loc[idx] == 1:
    #         data_df.drop([idx], inplace = True)
    #         data2_df.drop([idx], inplace = True)
    data_df.reset_index(inplace = True)
    data2_df.reset_index(inplace = True)
    data_df['avg_acc']=data_df['avg_acc'].rolling(smooth).mean()
    data2_df['avg_acc']=data2_df['avg_acc'].rolling(smooth).mean()
    data_df['acc_std_dev']=data_df['acc_std_dev'].rolling(smooth).mean()
    data2_df['acc_std_dev']=data2_df['acc_std_dev'].rolling(smooth).mean()
    #plot data
    fig, axes = plt.subplots(nrows=2,  ncols=1)
    data_df['avg_acc'].dropna().plot(kind='line',y=1, use_index=True, color='r',legend = True, ax=axes[0])
    data2_df['avg_acc'].dropna().plot(kind='line',y=1, use_index=True, color='b', legend = True, ax=axes[0])
    # ax2=ax.twinx()
    data_df['acc_std_dev'].dropna().plot(kind='line',y=1, use_index=True, color='r',legend = True, ax=axes[1])
    data2_df['acc_std_dev'].dropna().plot(kind='line',y=1, use_index=True, color='b',legend = True, ax=axes[1])

    axes[0].title.set_text("Average Acceleration m/s^2")
    axes[0].set_ylabel("avg acceleration")
    axes[1].title.set_text("Standard Deviation of Acceleration")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("std deviation")
    axes[0].legend(["no 2ndPF","2ndPF"])
    axes[1].legend(["no 2ndPF","2ndPF"])


fig.tight_layout(pad=0.5)
plt.show()

#save plot