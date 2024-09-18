import os
import glob
import numpy as np


def load_data_for_keras(dd='.\\figs', START=1, NUM_IMAGES=1000, TSTEP=1):

    Xs = list()               # Training inputs [0..TEST_IMAGES-1]
    Ys = list()               # Training outputs [1..TEST_IMAGES]
    Ysdates = list()
    ff = sorted(glob.glob(dd+'\\*.npy'))
    # XXX: Load the first TEST_IMAGES for training
    # print('In load image!')
    for i in range(START, START+NUM_IMAGES):
        # print('i is: ', i)
        for j in range(TSTEP):
            # print('j is: ', j)
            img = np.load(ff[i+j])   # PIL image
            np.all((img > 0) & (img <= 1))
            # print('loaded i, j: X(i+j)', i, j,
            #       ff[i+j].split('/')[-1].split('.')[0])
            Xs += [img]

        # XXX: Just one output image to compare against
        # XXX: Now do the same thing for the output label image
        # img = Image.open(ff[i+TSTEP]).convert('LA')
        img = np.load(ff[i+TSTEP])   # PIL image
        np.all((img > 0) & (img <= 1))
        # print('loaded Y: i, TSTEP: (i+TSTEP)', i, TSTEP,
        #       ff[(i+TSTEP)].split('/')[-1].split('_')[0])
        Ysdates.append(ff[(i+TSTEP)].split('\\')[-1].split('.')[0])
        Ys += [img]

    # XXX: Convert the lists to np.array
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    np.expand_dims(Xs, axis=-1)
    np.expand_dims(Ys, axis=-1)
    return Xs, Ys, Ysdates

def build_keras_model(shape, inner_filters, LR=1e-3):
    inp = Input(shape=shape[1:])
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(7, 7),
        padding="same",
        data_format='channels_last',
        activation='relu',
        return_sequences=True)(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        data_format='channels_last',
        padding='same',
        activation='relu',
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=inner_filters,
        kernel_size=(3, 3),
        data_format='channels_last',
        padding='same',
        activation='relu',
        return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=inner_filters,
        kernel_size=(1, 1),
        data_format='channels_last',
        padding='same',
        activation='relu',
        return_sequences=True)(x)
    # XXX: 3D layer for images, 1 for each timestep
    x = Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="relu",
        padding="same")(x)
    # XXX: Take the average in depth
    x = keras.layers.AveragePooling3D(pool_size=(shape[1], 1, 1),
                                      padding='same',
                                      data_format='channels_last')(x)
    # XXX: Flatten the output
    # x = keras.layers.Flatten()(x)
    # # XXX: Dense layer for 1 output image
    # tot = 1
    # for i in shape[2:]:
    #     tot *= i
    # print('TOT:', tot)
    # x = keras.layers.Dense(units=tot, activation='relu')(x)
    # # XXX: Reshape the output
    x = keras.layers.Reshape(shape[2:4])(x)

    # XXX: The complete model and compiled
    model = Model(inp, x)
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(learning_rate=LR),)
    return model


def keras_model_fit(model, trainX, trainY, valX, valY, batch_size):
    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10,
                                                   restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                  patience=5)

    # Define modifiable training hyperparameters.
    epochs = 10 
    # batch_size = 2

    # Fit the model to the training data.
    history = model.fit(
        trainX,                 # this is not a 5D tensor right now!
        trainY,                 # this is not a 5D tensor right now!
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(valX, valY),
        verbose=1,
        callbacks=[early_stopping, reduce_lr])
    # callbacks=[reduce_lr])
    return history
