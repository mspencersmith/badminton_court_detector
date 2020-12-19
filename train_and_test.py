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
lr = 0.000001
fact = 0.1
pat = 2
thr = 0.01
val_pct = 0.20 # reserves % of data for validation
bat_size = 150
lin_lay = 512


model_name = f"model-{int(start)}-lr{lr}-factor{fact}pat{pat}-thr{thr}-val_pct{val_pct}-bat_size{bat_size}-{img_wid}-{img_hei}-{lin_lay}"
file = f"bcf_models/{model_name}.pth"


data_set = np.load("bcf_data/data_set.npy", allow_pickle=True)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=fact, patience=pat, threshold=thr, verbose=True)
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

def _prep_batch(X, y, batch_size, train=False):
    acc_count, loss_count = 0, 0
    length = len(X)
    batches = length / batch_size
    for i in tqdm(range(0, length, batch_size)):
        batch_X = X[i:i+batch_size].view(-1, 1, img_wid, img_hei)
        batch_y = y[i:i+batch_size]
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        accuracy, loss = fwd_pass(batch_X, batch_y, train)
        acc_count += accuracy
        loss_count += loss
    acc_count = acc_count / batches
    loss_count = loss_count / batches
    return acc_count, loss_count

def test():
    test_batch = 10
    test_X, test_y = X[-val_size:], y[-val_size:]
    with torch.no_grad():
        val_acc, val_loss = _prep_batch(test_X, test_y, test_batch)
    
    print("\nOut of sample test")
    print(f"val_acc: {val_acc}, val_loss: {val_loss}")
    return val_acc, val_loss

def train(net, save=False):
    EPOCHS = 200
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    print(f"Training {model_name}, batch_size: {bat_size}, EPOCHS: {EPOCHS}")
    start_train = time.time()
    with open(f"bcf_logs/{model_name}.log", "a") as f:
        for epoch in range(EPOCHS):
            print(epoch)
            acc, loss = _prep_batch(train_X, train_y, bat_size, train=True)
            val_acc, val_loss = test()
            scheduler.step(val_loss)
            f.write(
                    f"{model_name},{round(time.time(),3)},{round(float(acc),4)},{round(float(loss),8)},{round(float(val_acc),4)},{round(float(val_loss),8)},{epoch}\n")
    
    finish_train = time.time()
    duration = round((finish_train - start_train)/60, 2)
    if save == True:
        torch.save(net.state_dict(), file)
        print(f"\n{model_name} took {duration} minutes to train and was saved in {file}")
    else:
        print(f"\n{model_name} took {duration} minutes to train.")

# train(net)
train(net, save=True)

finish = time.time()
mins = round((finish - start)/60, 2)
print(f"\nTotal time taken {mins} minutes.")
