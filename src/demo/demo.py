from keras.models import load_model
import numpy as np
import os
from data_set import DataSet
import matplotlib.pyplot as plt
from subprocess import call
from extractor import Extractor
import glob

data = DataSet()


def get_model():
    model = load_model(os.path.join('demo', 'pooling-features.046-2.029.hdf5'))
    return model


def get_frames_for_sample():
    images = sorted(glob.glob(os.path.join('demo', 'frames', 'frame' + '*jpg')))
    return images


def rescale_list(frames, size=40):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the original list."""
    assert len(frames) >= size

    # Get the number to skip between iterations.
    skip = len(frames) // size

    # Build our new output.
    output = [frames[i] for i in range(0, len(frames), skip)]

    # Cut off the last one if needed.
    return output[:size]


def extract_features():

    frames = get_frames_for_sample()
    frames = rescale_list(frames)

    model = Extractor()

    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    return np.array(sequence)


def predict_class_probabilities(features):

    features = features.reshape((1, features.shape[0], features.shape[1]))

    model = load_model(os.path.join('demo', 'pooling-features.046-2.029.hdf5'))
    prediction = model.predict(features, 1, 1)
    prediction = prediction.reshape((51,))

    top_5_indicies = np.argpartition(prediction, -5)[-5:]

    y_pos = np.arange(len(top_5_indicies))

    classes = []
    for elem in top_5_indicies:
        classes.append(data.classes[elem])

    plt.bar(y_pos, prediction[top_5_indicies])
    plt.xticks(y_pos, classes)
    plt.ylabel('Genauigkeit')

    plt.savefig(os.path.join('demo', 'prediction.png'))


def extract_frames(video_file):
    call(["ffmpeg", "-i", video_file, os.path.join('demo', 'frames', 'frames' + "frame-%04d.jpg")])

if __name__ == '__main__':
    video_file = os.path.join('data', 'hmdb', 'clap', '#20_Rhythm_clap_u_nm_np1_fr_goo_0.avi')

    extract_frames(video_file)
    features = extract_features()

    predict_class_probabilities(features)

