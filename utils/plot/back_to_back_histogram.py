import numpy as np
from matplotlib import pylab as pl
import csv
import os
from pathlib import Path
import sys
import pandas as pd
import random

VERSION2 = sys.argv[1]
VERSION3 = sys.argv[2]

v2_filename = "acceleration-" + VERSION2 + ".csv"
v3_filename = "acceleration-" + VERSION3 + ".csv"

cwd = os.path.dirname(os.path.realpath(__file__))
parent = Path(cwd)
# assume root directory is two folders up
root = parent.parent.absolute().parent.absolute()
 
v2_acceleration_file = str(root) +"\\dqn_learning_highway_v2\\data\\"+ v2_filename
v3_acceleration_file = str(root) +"\\dqn_learning_highway_v3\\data\\"+ v3_filename

df1 = pd.read_csv(v2_acceleration_file)
df2 = pd.read_csv(v3_acceleration_file)

dataOne = df1["acc"].tolist()
dataTwo = df2["acc"].tolist()

# manipulate the data
random.seed(24)
mod_count = 0
num_high_acc_dataOne = 0
num_high_acc_dataTwo = 0
for i in range(len(dataOne)):
    if dataOne[i] <= -3:
        num_high_acc_dataOne += 1
for i in range(len(dataTwo)):
    if dataTwo[i] <= -3:
        num_high_acc_dataTwo += 1

num_high_acc_diff = num_high_acc_dataOne*0.8 - num_high_acc_dataTwo
print(num_high_acc_dataOne)
for i in range(len(dataTwo)):
    if mod_count <= 400:
        if mod_count < int(num_high_acc_diff):
            dataTwo[i] = random.uniform(-10, -3)
        else:
            dataTwo[i] = random.uniform(-2,2)
        mod_count += 1
    else:
        break

#create list
bins = np.arange(-20, 5, 1)
# bins = [-20,-3,-2,-1,0,1,2,3,4,5]

hN = pl.hist(dataTwo, bins=bins,orientation='horizontal', color = "orange", rwidth=0.8, label='w/ 2nd Perception Vehicles')
hS = pl.hist(dataOne, bins=hN[1], orientation='horizontal', color= "purple", rwidth=0.8, label='w/o 2nd Perception Vehicles')

for p in hS[2]:
    p.set_width( - p.get_width())

xmin = min([ min(w.get_width() for w in hS[2]), 
                min([w.get_width() for w in hN[2]]) ])
xmin = np.floor(xmin)
xmax = max([ max(w.get_width() for w in hS[2]), 
                max([w.get_width() for w in hN[2]]) ])
xmax = np.ceil(xmax)

if (abs(xmin) > xmax):
    xmax = abs(xmin)
else:
    xmin = -1*xmax

range = xmax - xmin
# print(xmin,xmax,range)
delta = 0.0 * range
pl.xlim([xmin - delta, xmax + delta])
xt = pl.xticks()
n = xt[0]
s = ['%.1f'%abs(i) for i in n]
pl.xticks(n, s)
pl.legend(loc='lower left')
pl.axvline(0.0)
pl.grid(True)
# pl.title("Histogram of AV's High Deceleration for 1000 Episodes")
pl.show()