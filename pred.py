import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Input, Reshape, ConvLSTM1D, Flatten
from tensorflow.keras.models import Model
from sklearn import metrics 
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import pickle


# XXX: Moneyness Bounds inclusive
LM = 0.9
UM = 1.1
MSTEP = 0.00333

# XXX: Tau Bounds inclusive
LT = 14
UT = 366
TSTEP = 5                       # days

DAYS = 365

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

def conv_model(shape, LR=1e-3):
    model = tf.keras.Sequential()
        # Input shape: (samples, time steps, sequence length, features)
    model.add(ConvLSTM1D(filters=32, kernel_size=7, activation='relu', padding='same', data_format='channels_last', return_sequences=True,
                   input_shape=(shape[1:])))
    model.add(BatchNormalization())
    model.add(ConvLSTM1D(filters=64, kernel_size=5, activation='relu', padding='same', data_format='channels_last', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM1D(filters=64, kernel_size=3, activation='relu', padding='same', data_format='channels_last', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM1D(filters=64, kernel_size=1, activation='relu', padding='same', data_format='channels_last', return_sequences=False))
    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=(3), activation="relu", padding="same"))
    model.add(Reshape((shape[2], 1)))  # Reshape to match the target shape
    
    # XXX: The complete model and compiled
    model.compile(loss=tf.keras.losses.MSE,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LR))
    print(model.summary())

    return model

def lstm_model(shape, LR=1e-3):
    model = tf.keras.Sequential()
    model.add(Input(shape[1:]))
    model.add(LSTM(units=125, return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(LSTM(units=125, return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(LSTM(units=125, return_sequences=False, activation='relu'))
    model.add(Dense(shape[2]))
    model.compile(loss=tf.keras.losses.MSE,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LR))
    print(model.summary())
    return model

def build_keras_model(model_name, shape, LR=1e-3):
    if(model_name == 'conv'):
        return conv_model(shape, LR)
    elif(model_name == 'lstm'):
        return lstm_model(shape, LR)
    else:
        raise ValueError("Invalid model type")

def keras_model_fit(model, trainX, trainY, valX, valY, batch_size, model_name, TSTEPS, term):

    # Early stopping and reduce learning rate callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10,
                                                   restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  patience=5)

    # Define modifiable training hyperparameters.
    epochs = 500 

    # Fit the model to the training data.
    history = model.fit(
        trainX,                 
        trainY,                 
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(valX, valY),
        verbose=1,
        callbacks=[early_stopping, reduce_lr])
    
    if not os.path.exists('./ml_models'):
        os.makedirs('./ml_models')
        
    model.save('./ml_models/model_%s_ts_%s_term_%s_fig.keras' % (model_name, TSTEPS, term))
    # Summarise history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    if not os.path.exists('./ml_models/history'):
        os.makedirs('./ml_models/history')
        
    plt.savefig('./ml_models/history/history_model_%s_ts_%s_term_%s_fig.pdf' % (model_name, TSTEPS, term))

    Yout = model(valX, training=False).numpy()
    # Save the metrics to a csv file
    
    if not os.path.exists('./ml_models/metrics'):
        os.makedirs('./ml_models/metrics')
        
    # XXX: Calculate the MSE, MAPE, R2
    mse = metrics.mean_squared_error(valY.reshape(valY.shape[0:2]), Yout.reshape(valY.shape[0:2]), multioutput='raw_values')
    mape = metrics.mean_absolute_percentage_error(valY.reshape(valY.shape[0:2]), Yout.reshape(valY.shape[0:2]), multioutput='raw_values') 
    r2 = metrics.r2_score(valY.reshape(valY.shape[0:2]), Yout.reshape(valY.shape[0:2]), multioutput='raw_values')

    with open('./ml_models/metrics/metrics_model_%s_ts_%s_term_%s_fig.csv' % (model_name, TSTEPS, term), 'w') as f:
        f.write("MSE, MAPE, R2, MSEstd, MAPEstd, R2std\n")
        f.write("%s, %s, %s, %s, %s, %s\n" % (np.mean(mse), np.mean(mape), np.mean(r2), np.std(mse), np.std(mape), np.std(r2)))

    return model 

def plot_predicted_outputs(valY, valOutY, Dates):

    # XXX: The moneyness
    MS = np.arange(LM, UM+MSTEP, MSTEP)

    for y, yp, yd in zip(valY, valOutY, Dates):
        y *= 100
        yp *= 100
        ynum = date_to_num(yd)
        pyd = num_to_date(ynum-1)
        fig, axs = plt.subplots(1, 3, figsize=(20, 9), subplot_kw={'ylim': (0, 40)})
        axs[0].title.set_text('Truth: ' + yd)
        # XXX: Make the y dataframe
        ydf = list()
        for cm, m in enumerate(MS):
            ydf.append([m, y[cm]])
        ydf = np.array(ydf)
        axs[0].plot(ydf[:, 0], ydf[:, 1], linewidth=0.1)
        axs[0].set_xlabel('Moneyness')
        axs[0].set_ylabel('Vol%')
        axs[1].title.set_text('Predicted: ' + yd)
        ypdf = list()
        for cm, m in enumerate(MS):
            ypdf.append([m, yp[cm]])
        ypdf = np.array(ypdf)
        axs[1].plot(ypdf[:, 0], ypdf[:, 1], linewidth=0.2)
        axs[1].set_xlabel('Moneyness')
        axs[1].set_ylabel('Vol %')

        # XXX: Previous day' volatility
        ximg = load_image(ynum-1)
        ximg = ximg[0:,[0]]
        ximg = ximg.reshape((ximg.shape[0],))
        ximg *= 100
        xdf = list()
        for cm, m in enumerate(MS):
            xdf.append([m, ximg[cm]])
        xdf = np.array(xdf)
        axs[2].plot(xdf[:, 0], xdf[:, 1], linewidth=0.2)
        axs[2].set_xlabel('Moneyness')
        axs[2].set_ylabel('Vol %')
        axs[2].title.set_text(pyd)
        plt.show()
        plt.close(fig)

def plot_predicted_outputs_reg(valY, valOutY):

    # XXX: The moneyness
    MS = np.arange(LM, UM+MSTEP, MSTEP)

    # XXX: Reshape the outputs
    valY = valY.reshape(valY.shape[0], len(MS))
    valOutY = valOutY.reshape(valOutY.shape[0], len(MS))

    print(valY.shape, valOutY.shape)

    for i in range(valY.shape[0]):
        y = valY[i]*100
        yp = valOutY[i]*100
        fig, axs = plt.subplots(1, 2)
        axs[0].title.set_text('Truth')

        # XXX: Make the y dataframe
        ydf = list()
        for cm, m in enumerate(MS):
            ydf.append([m, y[cm]])

        ydf = np.array(ydf)
        axs[0].plot(ydf[:, 0], ydf[:, 1],
                            linewidth=0.2,
                            antialiased=True)
        axs[0].set_xlabel('Moneyness')
        axs[0].set_ylabel('Vol %')
        axs[1].title.set_text('Predicted')
        ypdf = list()
        for cm, m in enumerate(MS):
            ypdf.append([m, yp[cm]])

        ypdf = np.array(ypdf)
        axs[1].plot(ypdf[:, 0], ypdf[:, 1],
                            linewidth=0.2,
                            antialiased=True)
        axs[1].set_xlabel('Moneyness')
        axs[1].set_ylabel('Vol %')

        plt.show()
        plt.close()

def regression_predict(dataDir, model="Ridge", TSTEPS=10, data = None, term=14):

    intercept = True 

    if(model == "Ridge"):
        treg = 'ridge'
        reg = Ridge(fit_intercept=intercept, alpha=1.0)
    elif(model == "Lasso"):
        treg = 'lasso'
        reg = Lasso(fit_intercept=intercept, alpha=0.001, max_iter=100000, selection='random')
    elif(model == "ElasticNet"):
        treg = 'enet'
        reg = ElasticNet(fit_intercept=intercept, alpha=0.001, selection='random')
    elif(model == "RandomForest"):
        treg = 'rf'
        reg = RandomForestRegressor(n_jobs=10, max_features='sqrt', n_estimators=150, bootstrap=True, verbose=1)
    elif(model == "XGBoost"):
        treg = 'xgb'
        reg = MultiOutputRegressor(
            xgb.XGBRegressor(tree_method='hist', multi_strategy='multi_output_tree', n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=10, verbosity=2)
        ) 
    elif(model == "OLS"):
        treg = 'ols'
        reg = LinearRegression(fit_intercept=intercept)
    else:
        raise ValueError("Invalid model type")

    # Load data
    if(data is None):
        trainX, trainY, valX, valY = load_train_test_split(TSTEPS=TSTEPS)
    else:
        trainX, trainY, valX, valY = data

    # XXX: Transform data 
    # We want each point in skew to be a feature and therefor estimate a coefficient for each point 
    # Flatten matrix
    print(trainX.shape, trainY.shape)
    print(valX.shape, valY.shape)
    trainX = trainX.reshape(trainX.shape[0], -1)
    trainY = trainY.reshape(trainY.shape[0], -1)
    valX = valX.reshape(valX.shape[0], -1)
    valY = valY.reshape(valY.shape[0], -1)
    print(trainX.shape, trainY.shape)
    print(valX.shape, valY.shape)

    # Fit the model
    reg.fit(trainX, trainY)


    # XXX: Predict on validation set
    vOutY = reg.predict(valX)

    # XXX: Calculate the MSE, MAPE, R2
    mse = metrics.mean_squared_error(valY, vOutY, multioutput='raw_values')
    mape = metrics.mean_absolute_percentage_error(valY, vOutY, multioutput='raw_values') 
    r2 = metrics.r2_score(valY, vOutY, multioutput='raw_values')

    # Print the mean metrics and their standard deviation
    print("STATS FOR ", model)
    print("Training score: ", reg.score(trainX, trainY))
    print("MSE mean: ", np.mean(mse), "MSE std: ", np.std(mse))
    print("MAPE mean: ", np.mean(mape), "MAPE std: ", np.std(mape))
    print("R2 score mean: ", np.mean(r2), "R2 std: ", np.std(r2))

    # XXX: Save the stats to a csv file
    if not os.path.exists('./reg_models/metrics'):
        os.makedirs('./reg_models/metrics')

    with open('./reg_models/metrics/metrics_model_%s_ts_%s_term_%s_fig.csv' % (treg, TSTEPS, term), 'w') as f:
        f.write("MSE, MAPE, R2, MSEstd, MAPEstd, R2std\n")
        f.write("%s, %s, %s, %s, %s, %s\n" % (np.mean(mse), np.mean(mape), np.mean(r2), np.std(mse), np.std(mape), np.std(r2)))

    import pickle
    with open('./reg_models/model_%s_ts_%s_term_%s_figs.pkl' % (treg, TSTEPS, term), 'wb') as f:
        pickle.dump(reg, f)


    # plot_predicted_outputs_reg(valY, vOutY)
    return (np.mean(mse), np.mean(mape), np.mean(r2))

def plot_grouped_bar(data):
    species = list(data.keys()) 

    stats = {
        "MSE": [data[sp][0] for sp in species],
        "MAPE": [data[sp][1] for sp in species],
        "R2": [data[sp][2] for sp in species]
    }

    x = np.arange(len(species))  # the label locations
    width = 0.35  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in stats.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Scores')
    ax.set_title('Scores by model and metric')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=4)

    plt.show()

def run_ml(models = ['conv', 'lstm'], TSTEPS = [5, 10, 20], terms = [31]):
    # Load the data
    for term in terms:
        for model_name in models:
            for tstep in TSTEPS:
                print("TSTEPS: ", tstep)
                data = load_train_test_split(TSTEPS=tstep, type=model_name, term=term)
                model = build_keras_model(model_name, data[0].shape)
                model = keras_model_fit(model, *data, 32, model_name, tstep, term)

def run_regression(models = [], TSTEPS = [5, 10, 20], terms=[31]):
    # Could make this whole process faster by loading the data once and then passing it to the functions
    # Dict to store the results
    for term in terms:
        for tstep in TSTEPS:
            print("TSTEPS: ", tstep)
            data = load_train_test_split(TSTEPS=tstep, term=term)
            for model in models:
                regression_predict("./data/figs/", model=model, TSTEPS=tstep, data=data, term=term)

        # for result in results:
        #     plot_grouped_bar(result)

def run_point(dd='./data/figs', model='Ridge', TSTEPS=10, term=31):
    # XXX: We will need to do steps 5, 10 and 20
    tX, tY, vX, vY = load_train_test_split(TSTEPS=TSTEPS, term=term, type='point')
    # tX = np.append(tX, vX, axis=0)
    # tY = np.append(tY, vY, axis=0)
    # print('tX, tY: ', tX.shape, tY.shape)

    # XXX: Validation set
    # print('vX, vY:', vX.shape, vY.shape)

    # Fill in NaN's... required for non-parametric regression
    # if dd == './gfigs':
    #     clean_data(tX, tY)
    #     clean_data(vX, vY)

    # XXX: Now go through the MS and TS
    mms = np.arange(LM, UM+MSTEP, MSTEP)

    count = 0
    for i, s in enumerate(mms):
        if count % 50 == 0:
            print('Done: ', count)
        count += 1
        # XXX: Make the vector for training
        k = np.array([s]*tX.shape[0]).reshape(tX.shape[0], 1)
        train_vec = np.append(tX[:, :, i], k, axis=1)
        # print(train_vec.shape, tY[:, i, j].shape)

        # XXX: Fit the ridge model
        treg = 'ridge'
        reg = Ridge(fit_intercept=True, alpha=1)
        reg.fit(train_vec, tY[:, i])
        # print('Train set R2: ', reg.score(train_vec, tY[:, i, j]))

        # XXX: Predict (Validation)
        # print(vX.shape, vY.shape)
        # k = np.array([s, t]*vX.shape[0]).reshape(vX.shape[0], 2)
        # val_vec = np.append(vX[:, :, i, j], k, axis=1)
        # vYP = reg.predict(val_vec)
        # vvY = vY[:, i, j]
        # r2sc = r2_score(vvY, vYP, multioutput='raw_values')
        # print('Test R2:', np.mean(r2sc))

        # XXX: Save the model
        import pickle
        with open('./point_models/pm_%s_ts_%s_p_%s_term_%s.pkl' %
                    (treg, TSTEPS, s, term), 'wb') as f:
            pickle.dump(reg, f)

def load_model(model_name, TSTEPS, term):
    if model_name == 'conv' or model_name == 'lstm':
        return tf.keras.models.load_model('./ml_models/model_%s_ts_%s_term_%s_fig.keras' % (model_name, TSTEPS, term))
    else:
        with open('./reg_models/model_%s_ts_%s_term_%s_figs.pkl' % (model_name, TSTEPS, term), 'rb') as f:
            return pickle.load(f)


if(__name__=="__main__"):
    TSTEPS = [5, 10, 20]
    for tstep in TSTEPS:
        run_point(TSTEPS=tstep)

    # TSTEPS = 5
    # data = load_train_test_split(TSTEPS=TSTEPS)
    # regression_predict(dataDir="./data/figs/", model="ElasticNet", TSTEPS=TSTEPS, data=data)
    # run_regression(["Lasso", "ElasticNet", "Ridge", "RandomForest", "XGBoost", "OLS"], TSTEPS=[5, 10, 20], terms=[31])