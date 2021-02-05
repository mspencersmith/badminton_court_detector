import matplotlib.pyplot as plt
import pandas as pd
import os.path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_1 = "detector/logs/model-1612398660-lr1e-06-factor0.1pat2-thr0.01-val_pct0.2-train_batch150-128-72-512-noSofm-Relu-mse.csv"
model_2 = "detector/logs/model-1611945391-lr1e-06-factor0.1pat2-thr0.01-val_pct0.2-train_batch150-128-72-512-sofm-bce.csv"

m1 = os.path.join(BASE_DIR, model_1)
m2 = os.path.join(BASE_DIR, model_2)

df1 = pd.read_csv(m1)
df2 = pd.read_csv(m2)

plt.style.use('seaborn')

fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)

ax1.set_title('Validation Set Accuracy')
ax1.plot(df1.epoch, df1.val_acc, label='MSE')
ax1.plot(df2.epoch, df2.val_acc, label='BCE')
ax1.legend()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')

ax2.set_title('Validation Set Loss')
ax2.plot(df1.epoch, df1.val_loss, label='MSE')
ax2.plot(df2.epoch, df2.val_loss, label='BCE')
ax2.legend()
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

plt.subplots_adjust(hspace=0.4)
plt.savefig('saves/loss_functions.png', dpi=400, bbox_inches='tight')
plt.show()