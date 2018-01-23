"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Reshape
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from data_set import DataSet
import time
import os.path
from keras.utils import plot_model

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=200):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', model + '-' + data_type + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        save_best_only=False)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'tensorboard_logs', 'train_lstm_trainiertes_inceptionv3'))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=15)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'tensorboard_logs', 'train_lstm_trainiertes_inceptionv3', 'lstm-training-' + str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = 100

    # Get data.
    X, y = data.get_all_sequences_in_memory('train', data_type)
    X_test, y_test = data.get_all_sequences_in_memory('test', data_type)


    # Get generators.
    generator = data.frame_generator(batch_size, 'train', data_type)
    val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Model.
    model = Sequential()

    # model.add(GlobalMaxPooling1D(input_shape=(seq_length, 2048)))

    # model.add(Flatten(input_shape=(seq_length, 2048)))
    # model.add(Reshape((seq_length * 2048, 1)))
    # model.add(MaxPooling1D(2, 2))
    # model.add(Flatten())

    # model.add(Conv1D(32, kernel_size=5))
    # model.add(Conv1D(1024, dropout=0.5))
    model.add(LSTM(2048, return_sequences=False,
                   input_shape=(seq_length, 2048),
                   dropout=0.5))
    # model.add(LSTM(2048, return_sequences=False,
    #                dropout=0.4))

    # model.add(Dropout(0.4))
    # model.add(Dense(2048))
    model.add(Dropout(0.4))
    model.add(Dense(2048))
    model.add(Dense(len(data.classes), activation='softmax'))

    # x = Dropout(0.4)(x)
    # x = Dense(1024, activation='relu')(x)

    # and a logistic layer
    # predictions = Dense(len(data.classes), activation='softmax')(x)

    # model = Sequential()
    # model.add(Flatten(input_shape=(seq_length, 2048)))
    # model.add(Dense(512))
    # model.add(Dropout(0.5))
    # model.add(Dense(512))
    # model.add(Dropout(0.5))
    # model.add(Dense(len(data.classes), activation='softmax'))

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                       metrics=['accuracy'])

    plot_model(model, to_file='data\\tensorboard_logs\\train_lstm_trainiertes_inceptionv3\\layers.png')

    print("summary: " + str(model.summary()))

    #print('LSTM summary:\n' + str(model.summary()))

    # Fit!
    # model.fit_generator(
    #     generator=generator,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=nb_epoch,
    #     verbose=1,
    #     callbacks=[tb, early_stopper, csv_logger, checkpointer],
    #     validation_data=val_generator,
    #     validation_steps=40,
    #     workers=4)

    model.fit(
        X,
        y,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[tb, csv_logger, checkpointer],
        epochs=nb_epoch)


def main():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'pooling'
    saved_model = None  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = 1000

    # Chose images or features and image shape based on network.
    data_type = 'features'
    image_shape = None

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
