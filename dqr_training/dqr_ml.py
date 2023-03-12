#! /apps/base/python3/bin/python
import act
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import time
import os
import numpy
import sys
from urllib.request import urlopen
import time
import pickle
import glob
from datetime import datetime
import random

ds='sgp30ebbrE13.b1'
varname='net_radiation'
vis_sdate='2022001'
vis_edate='20221231'
min_max=0
n_est=25
depth=3
ndays=3
max_bad=3
skip_dqrs=0
period=15
minp=5

def fileExist(ds, date):
    site = ds[:3]
    fdir = os.path.join('/data', 'archive', site, ds, ds + '*' + date + '*')
    result = glob.glob(fdir)
    if (len(result) == 0):
        result = 0
    else:
        result = 1
    return result

def visualizeResults(ds, varname, fig_name, score, fi):
    sdate = vis_sdate
    edate = vis_edate
    fi_name = ['data', 'std', 'max', 'min']
    print('Reading in: ', ds, ' ', varname, ' from ', sdate, '-', edate)
    data, flag_2d, sample_time = getTrainingSet(ds, varname, [sdate], [edate], 1)
    site = ds[:3]
    write_dir = os.path.join('./', 'results', site)
    ml_file = os.path.join(write_dir, ds + '_' + varname + '.p')

    with (open(ml_file, "rb")) as openfile:
        ml = pickle.load(openfile)

    result = ml.predict(data)

    nf = len(fi)
    fig, ax = plt.subplots(nrows=nf, ncols=1, figsize=(12, 12))
    nf = 1
    ms = 8.
    ls = 1.5
    sample_time = np.array(sample_time)
    for i in range(nf):
        ax[i].plot_date(np.reshape(sample_time, len(sample_time[0])), data[fi_name[i]], '-', linewidth=ls)
        ax[i].set_title(ds + ' ' + fi_name[i] + ' ' + varname + ' from ' + sdate + '-' + edate + ' FI:' + str(round(fi[i], 2)))

        if (i == 0):
            ax[i].annotate('Score=' + str(round(score, 2)), (sample_time[0, 0], max(data[fi_name[i]])))
        formatter = DateFormatter('%m/%d')
        plt.gcf().axes[i].xaxis.set_major_formatter(formatter)

        idx = (result == 0)
        index = np.where(idx)
        sample_data = np.array(data[fi_name[i]])
        new_time = np.reshape(sample_time[0, index], len(index[0]))
        new_data = sample_data[index]

        ax[i].plot_date(new_time, new_data, '.', color='r', markersize=ms)

    date = time.strftime("%Y%m%d")
    fdir = './images/' + date + '/'
    try:
        os.stat(fdir)
    except:
        os.mkdir(fdir)
    fig.tight_layout()
    fig.savefig(fdir + ds + '_' + varname + '_' + fig_name + '_' + sdate + '_' + edate + '.png')
   

def getTrainingSet(ds, varname, sdate, edate, visualize):
    train_data = []
    train_std = []
    train_min = []
    train_max = []
    train_time = []
    flag_2d = 0
    if (len([visualize]) == 0):
        visualize = 0
    site = ds[0:3]
    files = glob.glob(os.path.join('/data', 'archive', site, ds, ds + '*'))
    files.sort()
    dates = np.array([int(f.split('.')[-3]) for f in files])
    for i in range(len(sdate)):
        print('   ', sdate[i], '-', edate[i])
        if (visualize == 0):
            if (vis_sdate < sdate[i] < vis_edate):
                continue
            if (vis_sdate < edate[i] < vis_edate):
                edate[i]='20161231'
        idx = np.where((dates >= int(sdate[i])) & (dates <= int(edate[i])))[0]
        read_files  = list(np.array(files)[idx])
        try:
            obj = act.io.armfiles.read_netcdf(read_files)
        except:
           print('Error')
           continue

        obj = obj.dropna(dim='time')
        data = obj[varname].values
        stime = obj['time'].values
        print('      Success')
        mask1 = data <= -9999.
        try:
            data[mask1] = numpy.nan
        except:
            data[mask1] = 0
        mask = np.array(np.isnan(data) | np.isinf(data)  | pd.isnull(data))
        if (len(data.shape) > 1 ):
            std = np.std(data, axis=1)
            amin = np.nanmin(data, axis=1)
            amax = np.nanmax(data, axis=1)
            data = np.nanmean(data, axis=1)
            train_std = np.concatenate((train_std, std))
            train_min = np.concatenate((train_min, amin))
            train_max = np.concatenate((train_max, amax))
            flag_2d = 1.
        else:
            data = data[~mask]
            stime = stime[~mask]
        train_data += data.tolist()
        #train_time += stime.tolist()
        train_time.append(stime)
        #train_data.append(data.tolist())
        #train_time.append(stime.tolist())
        del data

    if len(train_std) == 0:
        data = np.transpose(train_data)
        df = pd.DataFrame(data=data, columns=['data'])
        train_std = df.rolling(period, min_periods=minp, center=True).std()
        train_min = df.rolling(period, min_periods=minp, center=True).min()
        train_max = df.rolling(period, min_periods=minp, center=True).max()

    df['std'] = train_std
    df['min'] = train_min
    df['max'] = train_max
    #train=[]
    #if (min_max == 0):
    #    train = np.vstack((np.transpose(train_data), np.transpose(train_std)))
    #else:
    #    train = np.vstack((np.transpose(train_data), np.transpose(train_std),
    #                       np.transpose(train_max), np.transpose(train_min)))

    del train_data
    del train_std
    del train_min
    del train_max

    return df, flag_2d, train_time

def getRandomDate():
    year = random.randint(2000, 2016)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    date = datetime(year, month, day).strftime("%Y%m%d")
    return date

if __name__ == '__main__':

    #####Get bad data sets based on DQR query####
    url = ''.join(("http://www.archive.arm.gov/dqrws/ARMDQR?datastream=",ds,"&varname=",varname))
    response = urlopen(url)
    timeblocks = []
    sdate = []
    edate = []
    ct=0
    if response.getcode() == 200: 
        for line in response.readlines():
            if (ct < skip_dqrs):
                ct+=1
                continue
            dummy = line.decode().replace('\n', '').split('|')
            sdate.append(time.strftime('%Y%m%d', time.gmtime(int(dummy[0]))))
            edate.append(time.strftime('%Y%m%d', time.gmtime(int(dummy[1]))))
            ct+=1
            if (ct > max_bad):
                break

    #####Get all DQR query####
    url = ''.join(("http://www.archive.arm.gov/dqrws/ARMDQR?datastream=", ds, "&varname=", varname, '&searchmetric=', 'incorrect,suspect,missing'))
    response = urlopen(url)
    all_dqr_sdate = []
    all_dqr_edate = []
    if response.getcode() == 200: 
        for line in response.readlines():
            dummy = line.decode().replace('\n', '').split('|')
            ed = time.strftime('%Y%m%d', time.gmtime(int(dummy[1])))
            if (float(ed) > 30000000.):
                continue
            all_dqr_sdate.append(time.strftime('%Y%m%d', time.gmtime(int(dummy[0]))))
            all_dqr_edate.append(ed)

    print('Getting Bad Data')
    bad_train, flag_2d, bad_time = getTrainingSet(ds, varname, sdate, edate, 0)
    blen = np.shape(bad_train)[0]
    if (blen == 0):
        blen = len(bad_train)
    blabel = np.full(blen, 0)

    #####Get Random Periods of Good Data#####
    print('Getting Good Data')
    #if flag_2d > 0:
    #   ndays=10
    print('Number of good days: ',ndays)
    good_sdate = []
    ct1 = 0
    ct2 = 0
    while (ct1 < ndays) and (ct2 < ndays*2):
        random_date = getRandomDate()
        ct2 += 1
        flag = 0
        for j in range(len(all_dqr_sdate)):
            if (all_dqr_sdate[j] < random_date < all_dqr_edate[j]):
                flag = 1
        if (flag == 1):
            continue     
        result=fileExist(ds, random_date)
        if (result == 0):
            continue
        good_sdate.append(random_date)
        ct1 += 1
   
    good_train, flag_2d, good_time = getTrainingSet(ds, varname, good_sdate, good_sdate, 0)

    glen = np.shape(good_train)[0]
    if (glen == 0):
        glen = len(good_train)
    glabel = np.full(glen, 1)

    label = np.concatenate((glabel, blabel))
    training = np.concatenate((good_train, bad_train), axis=0)

    del good_train
    del bad_train

    ###### Machine Learning Code#########
    ###### Can update to different ######
    ###### methods, nearest neighbor ####
    ###### is the simplist at the moment#####
    print('Training ML Algorithm....')
    X_train, X_test, y_train, y_test = train_test_split(training, label)
    #knn = KNeighborsClassifier(n_neighbors = 3)
    #if flag_2d > 0:
    #   n_est=15
    #   depth=3
    knn = RandomForestClassifier(max_depth=depth, random_state=0, n_estimators=n_est)
    knn.fit(X_train, y_train)

    score = knn.score(X_test, y_test)
    print(score)
    fi = knn.feature_importances_

    site = ds[:3]
    write_dir = os.path.join('./', 'results', site)
    write_file = os.path.join(write_dir, ds+'_'+varname+'.p')

    try:
        os.stat(write_dir)
    except:
        os.mkdir(write_dir)

    pickle.dump(knn,open(write_file,'wb'))

    fig_name = 'RF_nest' + str(n_est) + '_depth' + str(depth)
    visualizeResults(ds, varname, fig_name, score, fi)
