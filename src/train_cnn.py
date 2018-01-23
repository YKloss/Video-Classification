from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data_set import DataSet
import os

data = DataSet()

# Callback: Save the model.
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'checkpoints', 'cnn2.{epoch:03d}-{val_loss:.2f}.hdf5'), verbose=1)

# Callback: Stop when val_loss is not changing for a number of epochs.
early_stopper = EarlyStopping(monitor='val_loss', patience=15)

# Callback: TensorBoard.
tensorboard = TensorBoard(log_dir=os.path.join('data', 'tensorboard_logs', 'train_two_inceptions2'), histogram_freq=0,
                          write_graph=True,
                          write_images=True)


def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators

    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=50,
        epochs=nb_epoch,
        callbacks=callbacks)

    return model


def freeze_all_but_top(model):
    # Train only the top layers.
    for layer in model.layers[:-2]:
        layer.trainable = False

    # Compile the model.
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


def freeze_all_but_mid_and_top(model):
    # Train the the last two inception modules and the classification layers.
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # Recompile.
    # It's recommended to use a low training rate,
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model


def get_generators():
    # The ImageDataGenerator takes images and transforms them.
    # This is also a basic image preprocessing step.
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=40.,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(299, 299),
        batch_size=40,
        classes=data.classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'test'),
        target_size=(299, 299),
        batch_size=40,
        classes=data.classes,
        class_mode='categorical')

    return train_generator, validation_generator


def get_model(weights='imagenet'):
    # Get the Inception-v3 model from Google. Exclude the top.
    base_model = InceptionV3(weights=weights, include_top=False)

    # Add our own classification layers.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)                     # Dropout is needed to avoid overfitting.
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(data.classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def main():
    # Get the model (Inception-v3 with our own classification layers with random initialized weights).
    model = get_model()
    generators = get_generators()

    # Transfer Learning: Retrain the classification layers
    model = freeze_all_but_top(model)
    model = train_model(model, 5, generators,
                        [checkpointer, tensorboard])

    # Fine tuning: Train the last two inception modules
    model = freeze_all_but_mid_and_top(model)
    train_model(model, 1000, generators, [checkpointer, tensorboard])


if __name__ == '__main__':
    main()
