import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#graphing options
#1 = cumulative collision rate during train
option = 1

#read data
cwd = os.getcwd()
file = r"data\data-" + sys.argv[2] + ".csv"
file2 = r"data\data-" + sys.argv[3] + ".csv"
data_df = pd.read_csv(file)
data2_df = pd.read_csv(file2)

smooth = 10

if sys.argv[1] == "spd":
    #remove collison rows
    for idx in data_df.index.values:
        if data_df['coll'].loc[idx] == 1 or data2_df['coll'].loc[idx] == 1:
            data_df.drop([idx], inplace = True)
            data2_df.drop([idx], inplace = True)
    data_df.reset_index(inplace = True)
    data2_df.reset_index(inplace = True)
    data_df['avg_spd']=data_df['avg_spd'].rolling(smooth).mean()
    data2_df['avg_spd']=data2_df['avg_spd'].rolling(smooth).mean()
    data_df['spd_std_dev']=data_df['spd_std_dev'].rolling(smooth).mean()
    data2_df['spd_std_dev']=data2_df['spd_std_dev'].rolling(smooth).mean()
    #plot data
    fig, axes = plt.subplots(nrows=2,  ncols=1)
    data_df['avg_spd'].dropna().plot(kind='line',y=1, use_index=True, color='r',legend = True, ax=axes[0])
    data2_df['avg_spd'].dropna().plot(kind='line',y=1, use_index=True, color='b', legend = True, ax=axes[0])
    # ax2=ax.twinx()
    data_df['spd_std_dev'].dropna().plot(kind='line',y=1, use_index=True, color='r',legend = True, ax=axes[1])
    data2_df['spd_std_dev'].dropna().plot(kind='line',y=1, use_index=True, color='b',legend = True, ax=axes[1])

    axes[0].title.set_text("Average Speed m/s")
    axes[0].set_ylabel("avg speed")
    axes[1].title.set_text("Standard Deviation of Speed")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("std deviation")
    axes[0].legend(["no 2ndPF","2ndPF"])
    axes[1].legend(["no 2ndPF","2ndPF"])
else:
    #remove collison rows
    for idx in data_df.index.values:
        if data_df['coll'].loc[idx] == 1 or data2_df['coll'].loc[idx] == 1:
            data_df.drop([idx], inplace = True)
            data2_df.drop([idx], inplace = True)
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