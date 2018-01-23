"""
Train different models with the extracted features from video files.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, GlobalMaxPooling1D, Reshape
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from data_set import DataSet
import os.path
from keras.utils import plot_model


def get_lstm_model(output_length, seq_length):
    model = Sequential()

    model.add(LSTM(2048, return_sequences=False,
                   input_shape=(seq_length, 2048),
                   dropout=0.5))
    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(Dense(output_length, activation='softmax'))

    return model


def get_global_pooling_model(output_length, seq_length):
    model = Sequential()

    model.add(GlobalMaxPooling1D(input_shape=(seq_length, 2048)))
    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(Dense(output_length, activation='softmax'))

    return model


def get_pooling_model(output_length, seq_length):
    model = Sequential()

    # The input has to be reshaped first in order to work for the pooling layer
    model.add(Flatten(input_shape=(seq_length, 2048)))
    model.add(Reshape((seq_length * 2048, 1)))
    model.add(MaxPooling1D(2, 2))

    # Flatten the output from pooling for the classification layers
    model.add(Flatten())

    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(Dense(output_length, activation='softmax'))

    return model


def train(seq_length, model_name,
          class_limit=None, batch_size=32, nb_epoch=200, log_files=None):
    # Callback: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints',
                              model_name + '-' + 'features' + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        save_best_only=False)

    # Callback: TensorBoard
    tb = TensorBoard(log_dir=log_files)

    # Callback: Stop when val_loss is not changing for a number of epochs.
    early_stopper = EarlyStopping(patience=15)

    # Helper: Save results.
    # timestamp = time.time()
    # csv_logger = CSVLogger(
    #     os.path.join('data', 'tensorboard_logs', 'train_global_pooling', 'lstm-training-' + str(timestamp) + '.log'))

    data = DataSet(
        seq_length=seq_length,
        class_limit=class_limit
    )

    # Get data.
    X, y = data.get_all_sequences_in_memory('train', 'features')
    X_test, y_test = data.get_all_sequences_in_memory('test', 'features')

    # Get the Model.
    if model_name == 'lstm':
        model = get_lstm_model(len(data.classes), seq_length)
    elif model_name == 'global_pooling':
        model = get_global_pooling_model(len(data.classes), seq_length)
    elif model_name == 'pooling':
        model = get_pooling_model(len(data.classes), seq_length)
    else:
        raise Exception("No valid model name.")

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    # Optional: plot the model
    # plot_model(model, to_file='data\\tensorboard_logs\\train_lstm_trainiertes_inceptionv3\\layers.png')

    print("summary: " + str(model.summary()))

    model.fit(
        X,
        y,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[tb, checkpointer, early_stopper],
        epochs=nb_epoch)


def main():

    model = 'pooling'           # Can be 'lstm', 'pooling' or 'global_pooling'
    class_limit = None          # int, can be 1-51 or None
    seq_length = 40             # Number of frames per video
    batch_size = 32
    nb_epoch = 1000
    log_files = os.path.join('data', 'tensorboard_logs', 'train_pooling')

    train(seq_length, model,
          class_limit=class_limit, batch_size=batch_size, nb_epoch=nb_epoch,
          log_files=log_files)


if __name__ == '__main__':
    main()
