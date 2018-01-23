import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv(
    os.path.join('data', 'tensorboard_logs', 'train_two_inceptions', 'run_.-tag-val_loss.csv'))


plt.plot(df['Value'][:81])
plt.ylabel('Test-Kostenfunktion')
plt.xlabel('Epoche')
plt.show()
