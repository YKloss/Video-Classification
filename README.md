# Video Classification with Machine Learning

The goal of this project is to investigate Machine Learning models that are based on a CNN architecture to classify video data.
The HMDB51 dataset is used to evaluate the models. The model with the best performance is chosen in order to label videos
based on the classes from the HMDB51 dataset.

# Investigated Methods

- Inception-v3 as a feature extractor
- Fine-tuned Inception-v3 as a feature extractor
- LSTM model that takes the extracted features from videos as input
- GlobalMaxPooling model that takes the extracted features from videos as input
- MaxPooling model that takes the extracted features from videos as input

# Requiremenmts

Please make sure to install the required Python3 packages from src/requirements.txt.
Furthermore, create the directories _checkpoints, hmdb, sequences, tensorboard_logs, test_ and _train_ in the src/data/ folder.

Get the data from http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/ and extract the directories for all classes
in the src/data/hmdb/ directory.
