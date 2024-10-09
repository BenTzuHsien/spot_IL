from torch.utils.data import DataLoader
import torch, os
from torchvision import transforms
from spotdatasetloader import SPOTDataLoader
from FiveResNet18MLP5 import FiveResNet18MLP5
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

# Setup Destination
DATASET_INIRIAL_PATH = os.getcwd() + '/dataset_initial/'
TRAIN_PATH = DATASET_INIRIAL_PATH + 'train/'
TEST_PATH = DATASET_INIRIAL_PATH + 'test/'
GOAL_PATH = DATASET_INIRIAL_PATH + 'goal/'
LABEL_PATH = TRAIN_PATH + 'labels_radians.npy'
WEIGHT_PATH = os.getcwd() + '/weights/FiveResNet18MLP5_initial/lr1e-4_with_scaling_2'
TRAINING_LOSS_PATH = WEIGHT_PATH + 'training_losses.npy'
ACCURACIES_PATH = WEIGHT_PATH + 'accuracies.npy'
if not os.path.exists(WEIGHT_PATH):
    os.mkdir(WEIGHT_PATH)

# Start from beginning, use 0
CONTINUE = 0

model = FiveResNet18MLP5()

# Preprocess for images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if torch.cuda.is_available():
    
    train_dataset = SPOTDataLoader(
        root_dir = TRAIN_PATH,
        goal_folder = GOAL_PATH,
        labels_file = LABEL_PATH,
        transform = data_transforms
    )

    model.cuda()
    print('Cuda')

else:
    train_dataset = SPOTDataLoader(
        root_dir = TRAIN_PATH,
        goal_folder = GOAL_PATH,
        labels_file = LABEL_PATH,
        transform = data_transforms
    )
    print('cpu')

# Hyper Parameters
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
BATCH_SIZE = 16

# Training Parameters
epoch = CONTINUE + 1
training_loss = 1e6
training_total_loss = 0
training_losses = []   #[training_loss training_average_loss]
WEIGHT_SAVING_STEP = 10
LOSS_SCALE = 1e3

# Validation Parameter
NUM_FOLD = 5
NUM_FOLD_TRAIN_ITER = 1
k_fold = KFold(NUM_FOLD, shuffle=True)
TOLERANCE = 1e-3
accuracies = []   #[train_accuracy valid_accuracy]

if CONTINUE > 1:
    model.load_state_dict(torch.load(WEIGHT_PATH + 'epoch_' + str(CONTINUE) + '.pth'))
    print('Weight Loaded!')
    training_losses = list(np.load(TRAINING_LOSS_PATH))[:CONTINUE]
    accuracies = list(np.load(ACCURACIES_PATH))[:CONTINUE]
    print('Parameter Loaded!')

while training_loss > ((TOLERANCE ** 2) * LOSS_SCALE):
    for fold, (train_ids, valid_ids) in enumerate(k_fold.split(train_dataset)):
        # print(f'FOLD {fold}')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        valid_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=valid_subsampler)

        # Train Model
        model.train()
        running_loss = 0.0
        for iter in range(NUM_FOLD_TRAIN_ITER):
            
            for current_images, goal_images, labels in train_dataloader:
                
                optimizer.zero_grad()
                output = model(current_images, goal_images)
                loss = loss_fn(output.flatten(), labels.float()) * LOSS_SCALE

                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
        training_loss = running_loss / len(train_dataloader)

        # Moving Average
        training_total_loss += training_loss * 5
        training_average_loss = training_total_loss / (len(training_losses) + 5)
        training_total_loss = training_average_loss * (len(training_losses) + 1)

        # Save training loss
        training_losses.append([training_loss, training_average_loss])
        print(f'Epoch {epoch}, Loss: {training_loss:.6f}, Average Loss: {training_average_loss:.6f}', end='; ')
        np.save(TRAINING_LOSS_PATH, training_losses)

        if (epoch % WEIGHT_SAVING_STEP) == 0:
            torch.save(model.state_dict(), (WEIGHT_PATH + 'epoch_' + str(epoch) + '.pth'))
            print('Save Weights', end='; ')

        # Valid Model
        model.eval()
        with torch.no_grad():

            num_correct, num_total = 0, 0
            for current_images, goal_images, labels in train_dataloader:
                output = model(current_images, goal_images)
                for i in range(len(output)):
                    loss = abs(output[i] - labels[i]).item()
                    num_total += 1
                    if loss < TOLERANCE:
                        num_correct += 1
            train_accuracy = (num_correct / num_total) * 100

            num_correct, num_total = 0, 0
            for current_images, goal_images, labels in valid_dataloader:
                output = model(current_images, goal_images)
                for i in range(len(output)):
                    loss = abs(output[i] - labels[i]).item()
                    num_total += 1
                    if loss < TOLERANCE:
                        num_correct += 1
            valid_accuracy = (num_correct / num_total) * 100

            accuracies.append([train_accuracy, valid_accuracy])
            print(f'Train Accuracy {train_accuracy:.2f}%, Valid Accuracy: {valid_accuracy:.2f}%')
            np.save(ACCURACIES_PATH, accuracies)

        epoch += 1

        if training_loss <= ((TOLERANCE ** 2) * LOSS_SCALE):
            break

    # print('All Folds')

print('Finished Training')
epoch -= 1

# Save last weight
torch.save(model.state_dict(), (WEIGHT_PATH + 'epoch_' + str(epoch) + '.pth'))
print('Save Last Weights')

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