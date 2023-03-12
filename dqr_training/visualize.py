#! /apps/base/python3/bin/python
from dqo.arm_files import DataDateQuery as dquery
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import time
import os
import numpy
import sys
import pickle

fac='C1'
ds='sgpkazrge'+fac+'.a1'
ml_class='randomforest'
#ml_class='nn3'
varname=['reflectivity_copol']

sdate='20170101'
sdate='20171001'
edate='20171231'

sdate='20160426'
edate=sdate
print('Reading in: ',ds,' from ',sdate,'-',edate)
file_data=dquery(ds,variables=varname,sdate=sdate,edate=edate,suppress=True,
      path_root='/data/archive/')

sample_time=file_data.matplotlib_times

period=15
minp=5
var=file_data.get_variable(varname[0])

### Remove -9999 Data ####
mask1=var<=-9999
var[mask1]=numpy.nan

if (len(var.shape) > 1 ):
	var_std=np.std(var,axis=1)
	var=np.nanmean(var,axis=1)
#sample_time=sample_time[mask1]

### Remove Nan, Inf, and None Data ###

if (len(var_std) == 0):
   var_std=pd.rolling_std(pd.DataFrame(data=var),period,center=True,min_periods=minp).values
   mask=np.isnan(var_std) | np.isinf(var_std)  | pd.isnull(var_std)
   var_std[mask]=0.
   mask=np.isnan(var) | np.isinf(var)  | pd.isnull(var)
   var=var[~mask]
   sample_time=sample_time[~mask]

data=[]
data=np.vstack((np.transpose(var),np.transpose(var_std)))

site=ds[:3]
write_dir=os.path.join(os.environ['DQ_DATA'],'machine_learning',site)
ml_file=os.path.join(write_dir,ds+'_'+varname[0]+'.p')

with (open(ml_file, "rb")) as openfile:
   ml=pickle.load(openfile)

result=ml.predict(np.transpose(data))

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1)

ax1.plot_date(sample_time,var,'-',linewidth=0.5)
formatter = DateFormatter('%m/%d')
plt.gcf().axes[0].xaxis.set_major_formatter(formatter)

idx=(result == 0)
print(idx)
index=np.where(idx)
sample_time=np.array(sample_time)
sample_data=np.array(var)
sample_std=np.array(var_std)

new_time=sample_time[index]
new_data=sample_data[index]
new_std=sample_std[index]

ax1.plot_date(new_time,new_data,'.',color='r',linewidth=0.5)

ax2.plot_date(sample_time,var_std,'-',linewidth=0.5)
ax2.plot_date(new_time,new_std,'.',color='r',linewidth=0.5)
formatter = DateFormatter('%m/%d')
plt.gcf().axes[1].xaxis.set_major_formatter(formatter)

#ax.grid(True,zorder=5)
date=time.strftime("%Y%m%d")
fdir='/data/home/theisen/plot/machine_learning/'+date+'/'
try:
   os.stat(fdir)
except:
   os.mkdir(fdir)
fig.savefig(fdir+ds+'_'+varname[0]+'_'+ml_class+'.png')


