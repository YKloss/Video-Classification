from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from data_set import DataSet

data = DataSet()

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow_from_directory(
        os.path.join('data', 'train'),
        target_size=(299, 299),
        batch_size=5,
        classes=data.classes,
        class_mode='categorical',
        save_to_dir=os.path.join('spike', 'preview'),
        save_prefix='img',
        save_format='png'):

    i += 1
    if i > 0:
        break  # otherwise the generator would loop indefinitely



# def get_generators():
#     train_datagen = ImageDataGenerator(
#         rescale=1. / 255,
#         shear_range=0.2,
#         horizontal_flip=True,
#         rotation_range=10.,
#         width_shift_range=0.2,
#         height_shift_range=0.2)
#
#     test_datagen = ImageDataGenerator(rescale=1. / 255)
#
#     train_generator = train_datagen.flow_from_directory(
#         os.path.join('data', 'train'),
#         target_size=(299, 299),
#         batch_size=32,
#         classes=data.classes,
#         class_mode='categorical')
#
#     validation_generator = test_datagen.flow_from_directory(
#         os.path.join('data', 'test'),
#         target_size=(299, 299),
#         batch_size=32,
#         classes=data.classes,
#         class_mode='categorical')
#
#     return train_generator, validation_generator