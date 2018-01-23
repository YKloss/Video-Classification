from keras.models import load_model
import numpy as np
import os
from data_set import DataSet
import matplotlib.pyplot as plt

data = DataSet()

model = load_model(os.path.join('demo', 'pooling-features.046-2.029.hdf5'))

sequence_path = os.path.join('data', 'sequences', 'inceptionv3',
                             '#20_Rhythm_clap_u_nm_np1_fr_goo_0' + '-' + '40' + '-' + 'features' + '.npy')

if os.path.isfile(sequence_path):
    sequence = np.load(sequence_path)
    sequence = sequence.reshape((1, sequence.shape[0], sequence.shape[1]))
    print(sequence.shape)
else:
    raise Exception('Sequence file not found.')

pred = model.predict(sequence, 1, 1)
pred = pred.reshape((51,))
print(pred)
print(pred.shape)

ind = np.argpartition(pred, -5)[-5:]

print(data.classes)

y_pos = np.arange(len(ind))

classes = []
for elem in ind:
    classes.append(data.classes[elem])

plt.bar(y_pos, pred[ind])
plt.xticks(y_pos, classes)
plt.ylabel('Genauigkeit')

plt.savefig(os.path.join('demo', 'prediction.png'))
# plt.show()
