import matplotlib.pyplot as plt
import numpy as np
import os

DPI = 120
FIGURE_SIZE_PIXEL = [2490, 1490]
FIGURE_SIZE = [fsp / DPI for fsp in FIGURE_SIZE_PIXEL]

def plot_graph(training_losses, accuracies, weight_save_step, figure_path=None, start_plot=0, end_plot=0):

    if start_plot == end_plot:
        return
    
    # Fill with zero
    for i in range(start_plot):
        training_losses[i] = [0, 0]
        accuracies[i] = [0, 0]

    # Plot Training Loss
    training_loss = [data[0] for data in training_losses]
    average_loss = [data[1] for data in training_losses]

    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    plt.scatter(range(start_plot + 1, end_plot + 1), training_loss[start_plot:], color='blue', label='Training Loss')
    plt.plot(range(start_plot + 1, end_plot + 1), average_loss[start_plot:], color='cyan', linestyle='-', label='Average Training Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()

    lowest_loss = training_loss[0]
    for i in range(end_plot):

        if training_loss[i] < lowest_loss:
            lowest_loss = training_loss[i]

        if ((i + 1) % weight_save_step) == 0:
            plt.annotate(str(round(training_loss[i], 6)), xy=((i + 1), training_loss[i]))

    plt.annotate(str(round(training_loss[end_plot - 1], 6)), xy=(end_plot, training_loss[end_plot - 1]))

    plt.text(0, plt.gca().get_ylim()[1], f'Lowest Loss: {lowest_loss: .6f}')

    if figure_path is not None:
        plt.savefig(figure_path + 'Training_loss.png')
        plt.close()

    else:
        plt.show()

    # Plot Accuracy
    train_accuracy = [data for data in accuracies]

    plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    plt.plot(range(start_plot + 1, end_plot + 1), train_accuracy[start_plot:], color='blue', linestyle='-', marker='o', label='Training Accuracy')
    plt.title(f"Accuracy")
    plt.xlabel("Epoches")
    plt.ylabel("Acurracy (%)")
    plt.legend()

    for i in range(end_plot):
        if ((i + 1) % weight_save_step) == 0:
            plt.annotate(str(round(train_accuracy[i], 2)), xy=((i + 1), train_accuracy[i]))
    plt.annotate(str(round(train_accuracy[end_plot - 1], 2)), xy=(end_plot, train_accuracy[end_plot - 1]))

    if figure_path is not None:
        plt.savefig(figure_path + 'Accuracy.png')
        plt.close()
    
    else:
        plt.show()

if __name__ == '__main__':
    WEIGHT_PATH = os.getcwd() + '/weights/FiveResNet18MLP5_mixed/lr1e-5_with_scaling/'
    
    # hyper_params_path = WEIGHT_PATH + 'hyper_params.npz'
    # loaded_params = np.load(hyper_params_path)
    # params_dict = {key: loaded_params[key].item() for key in loaded_params}
    # print(params_dict)

    # NUM_FOLD = 5
    # END_PLOT = 0
    # START_PLOT = 0

    # for i in range(NUM_FOLD):
    #     fold_path = WEIGHT_PATH + 'fold_' + str(i) + '/'
    #     TRAINING_LOSSES_PATH = fold_path + 'training_losses.npy'
    #     ACCURACIES_PATH = fold_path + 'accuracies.npy'

    #     END_PLOT = len(np.load(ACCURACIES_PATH))
    #     training_losses = list(np.load(TRAINING_LOSSES_PATH))[:END_PLOT]
    #     accuracies = list(np.load(ACCURACIES_PATH))[:END_PLOT]
    #     # FIGURE_PATH = os.getcwd() + '/Results/FiveResNet18MLP5_mixed/lr1e-5_with_scaling/'
    #     FIGURE_PATH = None

    #     plot_graph(training_losses, accuracies, figure_path=FIGURE_PATH, fold=i, start_plot=START_PLOT, end_plot=END_PLOT)