import matplotlib.pyplot as plt
import pandas as pd

model_name = "logs/model-1608419488-lr1e-06-factor0.1pat2-thr0.01-val_pct0.2-bat_size150-128-72-512.csv"

df = pd.read_csv(model_name)
print(df)

plt.style.use('seaborn')

fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)

ax1.set_title('Training Set Accuracy vs Validation Set Accuracy')
ax1.plot(df.epoch, df.acc, label='Training')
ax1.plot(df.epoch, df.val_acc, label='Validation')
ax1.legend()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')

ax2.set_title('Training Set Loss vs Validation Set Loss')
ax2.plot(df.epoch, df.loss, label='Training')
ax2.plot(df.epoch, df.val_loss, label='Validation')
ax2.legend()
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

plt.subplots_adjust(hspace=0.4)
plt.show()