"""
Train and test model here. Best to start without a scheduler to decrease
the learning rate until you have experimented with what value works best.
Try and keep validation % (val_pct) close to 20% to keep model honest. 
The larger your data set the more batches you will have to use.
"""
import numpy as np
import prep
import os.path
import time
import torch.optim as optim
import torch
import torch.nn as nn
from nn_model import Net
from pathlib import Path
from tqdm import tqdm

img_wid = 128
img_hei = 72

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

print("\nCreating neural network.. \n")
net = Net().to(device)

start = time.time()
lr = 0.000001
fact = 0.1
pat = 2
thr = 0.01
val_pct = 0.20 # reserves % of data for validation
train_batch = 150
test_batch = 10
lin_lay = 512
EPOCHS = 1

BASE_DIR = Path(__file__).resolve().parent.parent
model_name = f"model-{int(start)}-lr{lr}-factor{fact}pat{pat}-thr{thr}-val_pct{val_pct}-train_batch{train_batch}-{img_wid}-{img_hei}-{lin_lay}"
model = os.path.join(BASE_DIR, f"models/{model_name}.pth")

data_path = os.path.join(BASE_DIR, "data/data_set_test.npy")
data_set = np.load(data_path, allow_pickle=True)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=fact, patience=pat, threshold=thr, verbose=True)
loss_function = nn.MSELoss()

X = prep.arr([i[0] for i in data_set], img_wid, img_hei)
y = torch.Tensor([i[1] for i in data_set])

val_size = int(len(X)*val_pct)
print(f"\nValidation set: {val_size} images")

train_X, train_y  = X[:-val_size], y[:-val_size]
test_X, test_y = X[-val_size:], y[-val_size:]


def start_training(net, save=False):
    """Trains model with in sample data and saves model if required"""

    print(f"Training {model_name}, batch_size: {train_batch}, EPOCHS: {EPOCHS}")
    start_train = time.time()  
    train_and_test()
    finish_train = time.time()
    duration = round((finish_train - start_train)/60, 2)
    if save == True:
        torch.save(net.state_dict(), model)
        print(f"\n{model_name} took {duration} minutes to train and was saved in {model}")
    else:
        print(f"\n{model_name} took {duration} minutes to train.")

def train_and_test():
    """Trains and tests model, adjusts learning rate when required and logs results"""
    log_path = f'logs/{model_name}.log'
    with open(log_path, "a") as f:
        for epoch in range(EPOCHS):
            print(epoch)
            acc, loss = pass_batch(train_X, train_y, train_batch, train=True)
            val_acc, val_loss = test()
            scheduler.step(val_loss)
            f.write(
                    f"{model_name},{round(time.time(),3)},{round(float(acc),4)},{round(float(loss),8)},{round(float(val_acc),4)},{round(float(val_loss),8)},{epoch}\n")

def test():
    """Performs out of sample test"""
    with torch.no_grad():
        val_acc, val_loss = pass_batch(test_X, test_y, test_batch)
    print("\nOut of sample test")
    print(f"val_acc: {val_acc}, val_loss: {val_loss}")
    return val_acc, val_loss

def pass_batch(X, y, batch_size, train=False):
    """Batches dataset ready for forward pass"""
    acc_count, loss_count = 0, 0
    length = len(X)
    batches = length / batch_size
    for i in tqdm(range(0, length, batch_size)):
        batch_X = X[i:i+batch_size].view(-1, 1, img_wid, img_hei)
        batch_y = y[i:i+batch_size]
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        accuracy, loss = foreword_pass(batch_X, batch_y, train)
        acc_count += accuracy
        loss_count += loss
    acc_count = acc_count / batches
    loss_count = loss_count / batches
    return acc_count, loss_count

def foreword_pass(X, y, train=False):
    """Trains and/or tests network calculating accuracy and loss"""
    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

start_training(net)
# start_training(net, save=True)

finish = time.time()
mins = round((finish - start)/60, 2)
print(f"\nTotal time taken {mins} minutes.")
