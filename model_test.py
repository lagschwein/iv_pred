import os
import pickle
import numpy as np
import matplotlib.pyplot as plt 
import glob
import tensorflow as tf
import pred
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error

def date_to_num(date, dd='./data/figs'):
    """
    Convert the date to the number of the file
    """
    ff = sorted(glob.glob(dd+'/*.npy'))
    count = 0
    for i in ff:
        if i.split('/')[-1].split('.')[0] == date:
            break
        count += 1
    return count

def term_to_index(term):
    """
    Converts the desired term of the options contract to the index of the ivs figures
    """
    return int((term - 14)//5)

def num_to_date(num, dd='./data/figs'):
    """
    Converts the number of the file to the associated date
    """
    ff = sorted(glob.glob(dd+'/*.npy'))
    return ff[num].split('/')[-1].split('.')[0]


def load_image(num, dd='./data/figs'):
    ff = sorted(glob.glob(dd+'/*.npy'))
    img = np.load(ff[num])
    return img


def load_data_for_keras(dd='./data/figs', START=1, NUM_IMAGES=1000, TSTEP=1, term=14):
    term = term_to_index(term)
    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Ys = list()               # Training outputs [1..TEST_IMAGES]
    Ysdates = list()
    ff = sorted(glob.glob(dd+'/*.npy'))
    # XXX: Load the first TEST_IMAGES for training
    for i in range(START, START+NUM_IMAGES):
        for j in range(TSTEP):
            img = np.load(ff[i+j])   # PIL image
            np.all((img > 0) & (img <= 1))
            Xs += [img]

        # XXX: Just one output image to compare against
        # XXX: Now do the same thing for the output label image
        img = np.load(ff[i+TSTEP])   # PIL image
        np.all((img > 0) & (img <= 1))
        Ysdates.append(ff[(i+TSTEP)].split('/')[-1].split('.')[0])
        Ys += [img]

    # XXX: Convert the lists to np.array
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    # Only take the first channel corresponding to the 14 day expiration contracts
    Xs = Xs[:, :, [term]]
    Ys = Ys[:, :, [term]]
    return Xs, Ys, Ysdates



def load_train_test_split(NIMAGES1=4252, NIMAGES2=1000, TSTEPS=2, type='conv', term=14):
    # XXX: Important dates
    # train from 2002-02-08 to 2018-12-31 (4252 images)
    # Test from 2019-01 onwards (image 4253 onwards)

    START = 0

    trainX, trainY, _ = load_data_for_keras(START=START, NUM_IMAGES=NIMAGES1, TSTEP=TSTEPS, term=term)
    trainX = trainX.reshape(trainX.shape[0]//TSTEPS, TSTEPS, *trainX.shape[1:])

    START = NIMAGES1

    valX, valY, _ = load_data_for_keras(START=START, NUM_IMAGES=NIMAGES2, TSTEP=TSTEPS, term=term)
    valX = valX.reshape(valX.shape[0]//TSTEPS, TSTEPS, *valX.shape[1:])

    if(type == 'lstm' or type == 'point'):
        trainX = trainX.reshape(*trainX.shape[0:3])
        trainY = trainY.reshape(*trainY.shape[0:2])
        valX = valX.reshape(*valX.shape[0:3])
        valY = valY.reshape(*valY.shape[0:2])
    
    return trainX, trainY, valX, valY

def load_model(model='ridge', lag=2, term=31, point=None):
    if model == 'conv' or model == 'lstm':
        return tf.keras.models.load_model('./ml_models/model_%s_ts_%s_term_%s.keras' % (model, lag, term))
    elif model == 'pmodel':
        if point is not None:
            return pickle.load(open('./point_models/pm_ridge_ts_%s_p_%s_term_%s.pkl' % (lag, point, term), 'rb'))
        pass
    else:
        return pickle.load(open('./reg_models/model_%s_ts_%d_term_%s_figs.pkl' % (model, lag, term), 'rb'))

def plot_feature_hmap(vv, X, name):
    print(name)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = X
    for i in range(vv.shape[0]):
        im = ax.plot(xs, vv[i],
                             antialiased=False, linewidth=2)
    ax.set_xlabel('Moneyness')
    ax.legend(range(vv.shape[0]))
    plt.savefig(name, bbox_inches='tight')
    # plt.show()


def plot_feature_importance(coeff, lag=2):
    # calculate the average absolute coefficient value for each feature
    avg_abs_coeff = np.mean(np.abs(coeff), axis=0)

    # normalize coefficients
    norm_coeff = avg_abs_coeff/ np.sum(avg_abs_coeff)
    print(norm_coeff.shape)

    # Plot bar plot
    fig, ax = plt.subplots(layout='constrained', figsize=(15, 6))
    plt.bar(range(len(norm_coeff)), norm_coeff)
    plt.xlabel('Feature Index', labelpad=30)
    plt.ylabel('Normalized Average Absolute Coefficient')
    plt.title('Feature Importance in multi-output %s model with lag %d' % ('Ridge', lag))
    plt.xticks(range(0, len(norm_coeff), 5), range(0, len(norm_coeff), 5))
    # Plot extra tick to show days which correspond to np.arange(coeff.shape[0], coeff.shape[1], coeff.shape[0])
    sec = ax.secondary_xaxis(location=0)
    sec.set_xticks(np.arange(coeff.shape[0], coeff.shape[1]+1, coeff.shape[0]), ['\nDay ' + str(i) for i in range(1, (coeff.shape[1]//coeff.shape[0])+1, 1)])
    plt.show()
    # plt.figure(figsize=(10, 6))
    # for i in range(coeff.shape[0]):
    #     plt.plot(coeff[i], label='Feature Importance')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Feature Importance')
    # plt.title('Feature Importance')
    # # Split the ticks into coeff.shape[1]/lag intervals labeled day 1 to day n
    # plt.xticks(np.arange(coeff.shape[0], coeff.shape[1], coeff.shape[0]), [str(i) for i in range(1, coeff.shape[1]//coeff.shape[0], 1)])
    # plt.tight_layout()
    # plt.grid(True)
    # plt.set_cmap('winter')
    # plt.show()

def plot_metrics():
    import re
    import pandas as pd
    import glob
    # Loop through each directory reg_models/metrics and ml_models/metrics and load each csv file into a pandas dataframe by its timestamp
    # load metrics form csv
    timesteps = [5, 10, 20]
    # Get all the csv files in the directory
    path1 = './reg_models/metrics'
    path2 = './ml_models/metrics'
    all_files = glob.glob(os.path.join(path1 + '/*.csv'))
    all_files += glob.glob(os.path.join(path2 + '/*.csv'))

    all_metrics = []
    for step in timesteps:
        search = r'.*_%s.*\.csv' % step
        li = [] 
        for dir in all_files:
            if re.search(search, dir):
                # Get the model name
                model = re.search(r'model_(.*)_', dir).group(1)
                df = pd.read_csv(dir)
                df = df.assign(model=model)
                li.append(df)
        # Concatenate the dataframes
        frame = pd.concat(li, axis=0, ignore_index=True)
        all_metrics.append(frame)
    print(all_metrics[0].head(10))
    print(all_metrics[1].head(10))
    print(all_metrics[2].head(10))
    
    # Plot the metrics
    pass

def predict_model(model_name, model, valX, valY, lag=None, term=None):
    if model_name == 'conv' or model_name == 'lstm':
        if model_name == 'lstm':
            valX = valX.reshape(*valX.shape[0:3])
        print(model.summary())
        return model(valX, training=False).numpy()
    elif model_name == 'pmodel':
        # XXX: The output vector
        out = np.array([0.0]*(valY.shape[0]*valY.shape[1]))
        out = out.reshape(*valY.shape)
        # XXX: Now go through the MS and TS
        mms = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)
        for i, s in enumerate(mms):
            # XXX: Load the model
            model = load_model('pmodel', lag, term, s)
            # XXX: Now make the prediction
            k = np.array([s]*valX.shape[0]).reshape(valX.shape[0], 1)
            val_vec = np.append(valX[:, :, i], k, axis=1)
            out[:, i] = model.predict(val_vec).reshape(valX.shape[0])
        return out
    else:
        # For regression models input is flattened
        return model.predict(valX.reshape(valX.shape[0], -1))

def run_dm_test(models=["ridge", "rf", "ols", "lasso", "enet", "xgb", "lstm"], TSTEPS=[5, 10, 20], terms=[31]):

    from dieboldmarinio import dm_test
    # Round robin the models running the dieboldmarinio test and store all data in a pandas dataframe
    for term in terms:

        for tstep in TSTEPS:
            print("TSTEPS: ", tstep)
            data = load_train_test_split(TSTEPS=tstep, term=term)
            Y = data[3]

            # create csv file to store results 
            if not os.path.exists('./dm_test'):
                os.makedirs('./dm_test')

            with open('./dm_test/dm_test_tstep_%s_term_%s.csv' % (tstep, term), 'w') as f:
                for name in models:
                    f.write("%s, %s, " % (name, name+'_p'))
                f.write("\n")

            for i in range(len(models)):
                model_result = list()
                for j in range(i+1, len(models)):

                    # Load the models
                    model1 = load_model(models[i], tstep, term)
                    model2 = load_model(models[j], tstep, term)

                    Y1 = predict_model(models[i], model1, data[2], data[3], tstep, term)
                    Y2 = predict_model(models[j], model2, data[2], data[3], tstep, term)

                    # Reshape the outputs so that they have the same shape
                    Y = Y.reshape(Y.shape[0], Y.shape[1])
                    Y1 = Y1.reshape(Y1.shape[0], Y1.shape[1]) 
                    Y2 = Y2.reshape(Y2.shape[0], Y2.shape[1])

                    tstat, p = dm_test(Y, Y1, Y2)
                    model_result.append((tstat, p)) 
                    print("Model 1: ", models[i], "Model 2: ", models[j])
                    print("Tstat: ", tstat, "P-value: ", p)

                # prepend the model_result list with 0.0 until it has the same length as the model list
                while len(model_result) < len(models):
                    model_result.insert(0, (0.0, 0.0))

                # Save the results to a csv file
                if not os.path.exists('./dm_test'):
                    os.makedirs('./dm_test')

                with open('./dm_test/dm_test_tstep_%s_term_%s.csv' % (tstep, term), 'a') as f:
                    for tstat, p in model_result:
                        f.write("%s, %s, " % (tstat, p))
                    f.write("\n")

def main(model_name, TSTEP, term, dd='./data/figs', NIMAGES=1000, plot=True, get_features=False):

    # Important date for testing
    START = date_to_num('20190102')

    # XXX: Load the model
    model = load_model(model_name, TSTEP, term)

    # XXX: Load the data
    X, Y, dates = load_data_for_keras(START=START, NUM_IMAGES=NIMAGES, TSTEP=TSTEP, term=term)

    X = X.reshape(X.shape[0]//TSTEP, TSTEP, *X.shape[1:])

    if(model_name == 'lstm' or model_name == 'pmodel'):
        X = X.reshape(*X.shape[0:3])
        Y = Y.reshape(*Y.shape[0:2])

    # XXX: Predict the IV skew
    Ypred = predict_model(model_name, model, X, Y, TSTEP, term)

    # XXX: If the model is a regression model, plot the feature importance
    if get_features and model_name != 'conv' and model_name != 'lstm' and model_name != 'pmodel':
        # XXX: Moneyness
        MS = Y.shape[1]
        MONEYNESS = [0, MS//2, MS-1]

        if model_name == 'ridge':
            ws = model.coef_.reshape(MS, TSTEP, MS)
            # XXX: Just get the top 10 results
            X = np.arange(pred.LM, pred.UM+pred.MSTEP, pred.MSTEP)

            for i in MONEYNESS:
                wsurf = ws[i]  # 1 IV point
                if dd == './data/figs':
                    name = 'm_%s_%s_fm.pdf' % (X[i],
                                                TSTEP)
                else:
                    name = 'm_%s_t_%s_%s_gfigs_fm.pdf' % (
                        X[i],
                        TSTEP)
                plot_feature_hmap(wsurf, X, name)

    # Reshape output 
    Y = Y.reshape(Y.shape[0], Y.shape[1])

    if plot:
        pred.plot_predicted_outputs(Y, Ypred, dates)
        return None, None, None, Y, Ypred

    else:
        # XXX: calculate metrics
        rmse = root_mean_squared_error(Y, Ypred, multioutput='raw_values')
        mapes = mean_absolute_percentage_error(Y, Ypred, multioutput='raw_values')
        r2 = r2_score(Y, Ypred, multioutput='raw_values')

        print("RMSE mean: ", np.mean(rmse), "RMSE std: ", np.std(rmse))
        print("MAPE mean: ", np.mean(mapes), "MAPE std: ", np.std(mapes))
        print("R2 mean: ", np.mean(r2), "R2 std: ", np.std(r2))


        return rmse, mapes, r2, Y, Ypred 


def model_surf_v_point_model():
    import dieboldmarinio as dmtest
    import pred
    TTS = [5, 10, 20]
    # XXX: Only Ridge model(s)
    for t in TTS:
        _, _, r2, y, yp = main(plot=False, TSTEP=t,term=31, model_name="ridge")
        _, _, r2k, yk, ypk = main(plot=False, TSTEP=t,term=31, model_name="pmodel")
        print(ypk.shape, yp.shape)
        assert (np.array_equal(y, yk))
        # XXX: Now we can do Diebold mariano test
        try:
            dstat, pval = dmtest.dm_test(y, yp, ypk)
        except dmtest.ZeroVarianceException:
            dstat, pval = np.nan, np.nan
            # XXX: save the dstat and pvals
        with open('./%s_pmodel_v_model_fig.csv' %
                    (t), 'w') as f:
            f.write('#ridge, pmodel\n')
            f.write('dstat,pval\n')
            f.write('%s,%s\n' % (dstat, pval))
            f.write('r2ridge,r2pmodel\n')
            f.write('%s,%s' % (np.mean(r2), np.mean(r2k)))


if __name__ == '__main__':
    # main('ridge', 5, 31, plot=True, get_features=True)
    # lag = 10
    # model = load_model(model='enet', lag=lag)
    # plot_feature_importance(model.coef_, lag)
    model_surf_v_point_model()

    # plot_metrics()