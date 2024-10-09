import matplotlib.pyplot as plt
import numpy as np
import os

WEIGHT_PATH = os.getcwd() + '/weights/FiveResNet18MLP5_initial/'
TRAINING_LOSS_PATH = WEIGHT_PATH + 'training_losses.npy'
ACCURACIES_PATH = WEIGHT_PATH + 'accuracies.npy'
WEIGHT_SAVING_STEP = 10

CONTINUE = 200
epoch = CONTINUE
training_losses = list(np.load(TRAINING_LOSS_PATH))[:CONTINUE]
accuracies = list(np.load(ACCURACIES_PATH))[:CONTINUE]

# Plot Training Loss
training_loss = [data[0] for data in training_losses]
average_loss = [data[1] for data in training_losses]

plt.scatter(range(epoch), training_loss, color='blue', label='Training Loss')
plt.plot(range(epoch), average_loss, color='cyan', linestyle='-', label='Average Training Loss')
plt.title("Training Loss")
plt.xlabel("Epoches")
plt.ylabel("Loss (radians)")
plt.legend()

lowest_loss = training_loss[0]
for i in range(epoch):

    if training_loss[i] < lowest_loss:
        lowest_loss = training_loss[i]

    if (i % WEIGHT_SAVING_STEP) == 0:
        plt.annotate(str(round(training_loss[i], 6)), xy=(i, training_loss[i]))

plt.annotate(str(round(training_loss[epoch - 1], 6)), xy=(epoch, training_loss[epoch - 1]))

plt.text(0, plt.gca().get_ylim()[1], f'Lowest Loss: {lowest_loss: .6f}')

plt.show()

# Plot Accuracy
train_accuracy = [data[0] for data in accuracies]
valid_accuracy = [data[1] for data in accuracies]

plt.plot(range(epoch), train_accuracy, color='blue', linestyle='-', marker='o', label='Training Accuracy')
plt.plot(range(epoch), valid_accuracy, color='orange', linestyle='-', marker='o', label='Validation Accuracy')
plt.title("Accuracy")
plt.xlabel("Epoches")
plt.ylabel("Acurracy (%)")
plt.legend()

for i in range(epoch):
    if (i % WEIGHT_SAVING_STEP) == 0:
        plt.annotate(str(round(train_accuracy[i], 2)), xy=(i, train_accuracy[i]))
        plt.annotate(str(round(valid_accuracy[i], 2)), xy=(i, valid_accuracy[i]))
plt.annotate(str(round(train_accuracy[epoch - 1], 5)), xy=(epoch, train_accuracy[epoch - 1]))
plt.annotate(str(round(valid_accuracy[epoch - 1], 5)), xy=(epoch, valid_accuracy[epoch - 1]))

plt.show()