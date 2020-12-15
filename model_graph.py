import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

model_name = "model-1608042373-lr5e-06-factor0.1pat10-thr1e-06-val_pct0.2-batches150-128-72-128"


def create_acc_loss_graph(model_name):
    contents = open(f"bcf_logs/{model_name}.log", "r").read().split("\n")

    times = []
    
    accuracies = []
    losses = []

    val_accuracies = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accuracies.append(float(val_acc))
            val_losses.append(float(val_loss))


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0))

    ax1.set_title(model_name)
    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accuracies, label="val_acc")
    ax1.legend(loc=2)
    ax1.axis([times[0], times[-1], -1, 1])
    # ax1.axis([times[0], times[-1], -100, 100])
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)
    # ax2.axis([times[0], times[-1], 0, 0.003])
    ax2.axis([times[0], times[-1], 0, 2])
    plt.show()
    print(len(times))

create_acc_loss_graph(model_name)
