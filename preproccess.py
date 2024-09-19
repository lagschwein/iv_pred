# XXX: Preprocess script to convert the excel HistoricalOptionsData to data of implied volatility against strike price

import os
import glob 
import zipfile as zip
import fnmatch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# XXX: Moneyness Bounds inclusive
# This includes ITM, ATM and OTM options as specified by the paper doi: 10.1016/j.irfa.2024.103406
LM = 0.8
UM = 1.21
MSTEP = 0.00333

# XXX: Days to expiration 
# We will look at short term options which are classified as less than 90 days to expiration
# according to the paper doi: 10.1016/j.irfa.2024.103406
LT = 14 
UT = 90 
TSTEP = 5
DAYS = 365

def date_to_num(date, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.npy'))
    count = 0
    for i in ff:
        if i.split('/')[-1].split('.')[0] == date:
            break
        count += 1
    return count

def num_to_date(num, dd='./figs'):
    ff = sorted(glob.glob(dd+'/*.npy'))
    return ff[num].split('/')[-1].split('.')[0]

def save_data_to_npy(data, dd='./figs'):
    for k in data.keys():
        np.save('./figs/%s.npy' % (k), data[k])

def preprocess_ivs_df(dfs: dict):
    toret = dict()
    for k in dfs.keys():
        df = dfs[k]
        # XXX: First only get those that have volume > 0
        df = df[df['Volume'] > 0].reset_index(drop=True)
        # XXX: Make the log of K/UnderlyingPrice
        df['m'] = (df['Strike']/df['UnderlyingPrice'])
        # XXX: Moneyness is not too far away from ATM
        df = df[(df['m'] >= LM) & (df['m'] <= UM)]
        # XXX: Make the days to expiration
        df['Expiration'] = pd.to_datetime(df['Expiration'])
        df['DataDate'] = pd.to_datetime(df['DataDate'])
        df['tau'] = (df['Expiration'] - df['DataDate']).dt.days
        # XXX: only those that meet the lower and upper bounds of days to expiration 
        df = df[(df['tau'] >= LT) & (df['tau'] <= UT)]
        df['tau'] = df['tau']/DAYS
        df['m2'] = df['m']**2
        # df['tau2'] = df['tau']**2
        # df['mtau'] = df['m']*df['tau']

        # XXX: This is the final dataframe
        dff = df[['IV', 'm', 'tau', 'm2']]
        toret[k] = dff.reset_index(drop=True)
    return toret

def build_grid_and_images(df):
    # print('building grid and fitting')
    # XXX: Now fit a multi-variate linear regression to the dataset
    # XXX: Model used for interpolation is also outlined in the paper https://www.nber.org/system/files/working_papers/w5500/w5500.pdf
    # one for each day.
    df = dict(sorted(df.items()))
    fitted_dict = dict()
    grid = dict()
    scores = list()
    for k in df.keys():
        # print('doing key: ', k)
        y = df[k]['IV']
        X = df[k][['m', 'm2']]
        # print('fitting')
        reg = LinearRegression(n_jobs=-1).fit(X, y)
        # print('fitted')
        fitted_dict[k] = reg
        scores += [reg.score(X, y)]

        # XXX: Now make the grid
        ss = []
        mms = np.arange(LM, UM+MSTEP, MSTEP)
        # tts = [i/DAYS for i in range(TAU-14, TAU+14+TSTEP, TSTEP)]
        # print('making grid: ', len(mms), len(tts))
        for mm in mms:
            # for tt in tts:
            # XXX: Make the feature vector
            ss.append([mm, mm**2])

        grid[k] = pd.DataFrame(ss, columns=['m', 'm2'])
        # print('made grid and output')

    print("average fit score: ", sum(scores)/len(scores))
    # XXX: Now make the smooth ivs surface for each day
    iv_line = dict()
    for k in grid.keys():
        piv = fitted_dict[k].predict(grid[k])
        iv_line[k] = pd.DataFrame({'IV': piv,
                                       'm': grid[k]['m']})
        iv_line[k]['IV'] = iv_line[k]['IV'].clip(0.01, None)

    # plt.figure(figsize=(10, 10))
    # for k in iv_line.keys():
    #     plt.scatter(df[k]['m'], df[k]['IV'], label=k+'-actual')
    #     plt.plot(iv_line[k]['m'], iv_line[k]['IV'], label=k+'-smooth') 
    #     plt.legend()
    #     plt.show()
    
    # XXX: Plot the heatmap
    save_data_to_npy(iv_line)




def main(mdir, years, months, instrument, dfs: dict):
    ff = []
    for f in os.listdir(mdir):
        for y in years:
            for m in months:
                # XXX: Just get the year and month needed
                tosearch = "*_{y}_{m}*.zip".format(y=y, m=m)
                if fnmatch.fnmatch(f, tosearch):
                    ff += [f]
    # print(ff)
    # XXX: Read the csvs
    for f in ff:
        z = zip.ZipFile(mdir+f)
        ofs = [i for i in z.namelist() if 'options_' in i]
        # print(ofs)
        # assert (1 == 2)
        # XXX: Now read just the option data files
        for f in ofs:
            key = f.split(".csv")[0].split("_")[2]
            df = pd.read_csv(z.open(f))
            df = df[df['UnderlyingSymbol'] == instrument].reset_index(
                drop=True)
            dfs[key] = df

def excel_to_images():
    dir = '../HistoricalOptionsData/'
    years = [str(i) for i in range(2002, 2015)]
    # years = [2002]
    months = ['January', 'February',  'March', 'April', 'May', 'June', 'July',
              'August', 'September', 'October', 'November', 'December'
              ]
    # months = ['April']
    instrument = ["SPX"]
    dfs = dict()
    # XXX: The dictionary of all the dataframes with the requires
    # instrument ivs samples
    for i in instrument:
        # XXX: Load the excel files
        main(dir, years, months, i, dfs)

        # XXX: Now make ivs surface for each instrument
        df = preprocess_ivs_df(dfs)

        # XXX: Build the images
        build_grid_and_images(df)

if __name__ == '__main__':
    excel_to_images()