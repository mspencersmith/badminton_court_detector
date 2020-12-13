import numpy as np
import time
import torch.optim as optim
import torch
import torch.nn as nn
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
batch_size = 200
lin_lay = 128


model_name = f"model-{int(start)}-lr{lr}-factor{fact}pat{pat}-thr{thr}-val_pct{val_pct}-batches{batch_size}-{img_wid}-{img_hei}-{lin_lay}"
file = f"bcf_models/{model_name}.pth"


data_set = np.load("bcf_data/data_set.npy", allow_pickle=True)
optimizer = optim.Adam(net.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=fact, patience=pat, threshold=thr, verbose=True)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in data_set]).view(-1, 128, 72)
X = X/255.0 # scales images between 0 and 1
y = torch.Tensor([i[1] for i in data_set])

val_size = int(len(X)*val_pct)
print(f"\nValidation set: {val_size} images")

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

def test():
    test_batch = 10
    test_X, test_y = X[-val_size:], y[-val_size:]
    val_acc, val_loss = 0, 0
    for i in tqdm(range(0, len(test_X), test_batch)):
        with torch.no_grad():
            batch_X = test_X[i:i+test_batch].view(-1, 1, img_wid, img_hei)
            batch_y = test_y[i:i+test_batch]
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            bat_acc, bat_loss = fwd_pass(batch_X, batch_y)
            val_acc += bat_acc
            val_loss += bat_loss
    val_acc = val_acc / test_batch
    val_loss = val_loss / test_batch
    print("\nOut of sample test")
    print(f"val_acc: {val_acc}, val_loss: {val_loss}")
    return val_acc, val_loss

def train(net, save=False):
    EPOCHS = 10
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    print(f"Training {model_name}, batch_size: {batch_size}, EPOCHS: {EPOCHS}")
    start_train = time.time()
    with open(f"bcf_logs/{model_name}.log", "a") as f:
        for epoch in range(EPOCHS):
            print(epoch)
            acc, loss = 0, 0
            for i in tqdm(range(0, len(train_X), batch_size)):
                batch_X = train_X[i:i+batch_size].view(-1, 1, img_wid, img_hei)
                batch_y = train_y[i:i+batch_size]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                bat_acc, bat_loss = fwd_pass(batch_X, batch_y, train=True)
                acc += bat_acc
                loss += bat_loss

            acc = acc / batch_size
            loss = loss / batch_size
            val_acc, val_loss = test()
            # scheduler.step(val_loss)
            f.write(
                f"{model_name},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")
    
    finish_train = time.time()
    duration = round((finish_train - start_train)/60, 2)
    if save == True:
        torch.save(net.state_dict(), file)
        print(f"\n{model_name} took {duration} minutes to train and was saved in {file}")
    else:
        print(f"\n{model_name} took {duration} minutes to train.")

train(net)
# train(net, save=True)

finish = time.time()
mins = round((finish - start)/60, 2)
print(f"\nTotal time taken {mins} minutes.")