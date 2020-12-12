import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from nn_model import Net
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
lr = 0.000005
fact = 0.1
pat = 10
thr = 0.000001
val_pct = 0.20 # reserves % of data for validation
batch_size = 20
lin_lay = 128


MODEL_NAME = f"model-{int(start)}-lr{lr}-factor{fact}pat{pat}-thr{thr}-val_pct{val_pct}-batches{batch_size}-{img_wid}-{img_hei}-{lin_lay}"
file = f"bcf_models/{MODEL_NAME}.pth"


data_set = np.load("bcf_data/data_set.npy", allow_pickle=True)
optimizer = optim.Adam(net.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=fact, patience=pat, threshold=thr, verbose=True)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in data_set]).view(-1, 128, 72)
X = X/255.0 # scales images between 0 and 1
y = torch.Tensor([i[1] for i in data_set])

val_size = int(len(X)*val_pct)
print(f"\nValidation set: {val_size} images")

test_X = X[-val_size:]
test_y = y[-val_size:]

def fwd_pass(X, y, train=False):
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

def test(size=val_size):
    X, y = test_X[:size], test_y[:size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(
            X.view(-1, 1, img_wid, img_hei).to(device), y.to(device))
    print("\nOut of sample test")
    print(f"val_acc: {val_acc}, val_loss: {val_loss}")
    return val_acc, val_loss

def train(net, save=False):
    EPOCHS = 1
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    print(f"Training {MODEL_NAME}, BATCH_SIZE: {BATCH_SIZE}, EPOCHS: {EPOCHS}")
    start_train = time.time()
    with open(f"logs/{MODEL_NAME}.log", "a") as f:
        for epoch in range(EPOCHS):
            print(epoch)
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, img_wid, img_hei)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

            val_acc, val_loss = test()
            # scheduler.step(val_loss)
            f.write(
                f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")
    
    finish_train = time.time()
    duration = round((finish_train - start_train)/60, 2)
    if save == True:
        torch.save(net.state_dict(), file)
        print(f"\n{MODEL_NAME} took {duration} minutes to train and was saved in {file}")
    else:
        print(f"\n{MODEL_NAME} took {duration} minutes to train.")

train(net)
# train(net, save=True)

finish = time.time()
mins = round((finish - start)/60, 2)
print(f"\nTotal time taken {mins} minutes.")