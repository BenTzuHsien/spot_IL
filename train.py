from torch.utils.data import DataLoader
import torch, os
from torchvision import transforms
from spotdatasetloader import SPOTDataLoader
from FiveResNet18MLP5 import FiveResNet18MLP5
import matplotlib.pyplot as plt
import numpy as np

# Setup Destination
DATASET_INIRIAL_PATH = os.getcwd() + '/dataset_initial/'
TRAIN_PATH = DATASET_INIRIAL_PATH + 'train/'
TEST_PATH = DATASET_INIRIAL_PATH + 'test/'
GOAL_PATH = DATASET_INIRIAL_PATH + 'goal/'
WEIGHT_PATH = os.getcwd() + '/weights/'
CONTINUE = 0   # Start from beginning, use 0

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = FiveResNet18MLP5()

if torch.cuda.is_available():
    
    train_dataset = SPOTDataLoader(
        root_dir=TRAIN_PATH,
        goal_folder=GOAL_PATH,
        labels_file=os.path.join(TRAIN_PATH, 'labels.npy'),
        transform=data_transforms
    )

    model.cuda()
    print('Cuda')

else:
    train_dataset = SPOTDataLoader(
        root_dir=TRAIN_PATH,
        goal_folder=GOAL_PATH,
        labels_file=os.path.join(TRAIN_PATH, 'labels.npy'),
        transform=data_transforms
    )
    print('cpu')

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

if CONTINUE > 1:
    model.load_state_dict(torch.load(WEIGHT_PATH + 'epoch_' + str(CONTINUE) + '.pth'))
    print('Weight Loaded!')

# Training Parameters
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

epoch = CONTINUE + 1
epoch_loss = 1
epoch_losses = []
patch = 0
NUM_PATCH = len(train_dataloader)
# if CONTINUE > 1:
#     epoch_losses = list(np.load(WEIGHT_PATH + 'epoch_losses.npy'))[:CONTINUE]

while epoch_loss > 1e-4:
    model.train()
    running_loss = 0.0
    valid_loss = 0.0
    for current_images, goal_images, labels in train_dataloader:

        optimizer.zero_grad()

        output = model(current_images, goal_images)
        loss = loss_fn(output, labels.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    epoch_losses.append(epoch_loss)

    print(f'Epoch {epoch}, Loss: {epoch_loss:.6f}')
    np.save(WEIGHT_PATH + 'epoch_losses.npy', epoch_losses)

    if (epoch % 20) == 0:
        torch.save(model.state_dict(), (WEIGHT_PATH + 'epoch_' + str(epoch) + '.pth'))
        print('Save Weights')
    
    epoch += 1

print('Finished Training')
epoch -= 1
torch.save(model.state_dict(), (WEIGHT_PATH + 'epoch_' + str(epoch) + '.pth'))
print('Save Weights')
epoch -= 1
plt.plot(epoch_losses)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Epoch Loss")

lowest_loss = epoch_losses[0]
for i in range(len(epoch_losses)):

    if epoch_losses[i] < lowest_loss:
        lowest_loss = epoch_losses[i]

    if (i % 20) == 0:
        plt.annotate(str(round(epoch_losses[i], 5)), xy=(i, epoch_losses[i]))

plt.annotate(str(round(epoch_losses[epoch], 5)), xy=(epoch, epoch_losses[epoch]))


plt.text(0, plt.gca().get_ylim()[1] - 0.05, f'Lowest Loss: {lowest_loss}')

plt.show()