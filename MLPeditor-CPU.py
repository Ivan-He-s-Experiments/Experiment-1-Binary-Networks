import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, TextBox
import pickle
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
import random

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    history = []
    class MLP(torch.nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = torch.nn.Linear(784, 128)
            self.fc1_1 = torch.nn.Linear(128, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 10)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc1_1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    # Load model and weights
    model1 = MLP()
    model1.load_state_dict(torch.load('rl_binary_b512_20ep_2lamb.pt'))
    model1 = model1
    model1.eval()
    layers = [model1.fc1,model1.fc1_1,model1.fc2,model1.fc3]
    layer = layers[0]
    # Extract weights and create a modifiable copy
    #weights = layer.weight.cpu().detach().numpy().copy()
    #print(torch.cdist(model.fc1.weight, model1.fc1.weight))
    weights = layer.weight.cpu().detach().numpy().copy()
    original = layer.weight.cpu().detach().numpy().copy()

    # Create figure and axes
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.2)



    # Plot weight matrix
    im = ax.imshow(weights, cmap='viridis', aspect='auto', origin='upper')
    plt.colorbar(im, ax=ax)


    # Text annotation for hover
    hover_text = ax.text(0.05, 1.05, '', transform=ax.transAxes, fontsize=10)

    accuracy_text = ax.text(0.30, 1.05, '', transform=ax.transAxes, fontsize=10)
    accuracy_num_texts1 = [ax.text(0.05 + n * 0.1, 1.13, f'{n}', transform=ax.transAxes, fontsize=10) for n in range(10)]
    accuracy_num_texts = [ax.text(0.05+n*0.1, 1.1, 'nan', transform=ax.transAxes, fontsize=10) for n in range(10)]

    # Selected region storage
    selected_region = None


    # Rectangle selector callback
    def on_select(eclick, erelease):
        global selected_region
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        # Convert to matrix indices
        j_min = max(0, int(np.floor(x_min + 0.5)))
        j_max = min(weights.shape[1], int(np.ceil(x_max - 0.5)))+1
        i_min = max(0, int(np.floor(y_min + 0.5)))
        i_max = min(weights.shape[0], int(np.ceil(y_max - 0.5)))+1

        selected_region = (i_min, i_max, j_min, j_max)
        print(f"Selected region: rows {i_min}-{i_max}, cols {j_min}-{j_max}")


    # Create rectangle selector
    rs = RectangleSelector(ax, on_select, useblit=True,
                           button=[1], minspanx=1, minspany=1,
                           spancoords='data', interactive=True)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    batch_size = 512
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    file = "all_history.pkl"

    eval_cnt = len(history)
    if os.path.exists(file) and eval_cnt == 0:
        with open(file, "rb") as fr:
            history = pickle.load(fr)
            eval_cnt = len(history)
    o_accuracy=None
    o_accuracy_per_class = None
    events_cnt = 0
    #events_history1 = [(weights.copy(), accuracy_text, accuracy_num_texts)]

    # Button callbacks
    def round_callback(event):
        global selected_region, weights, events_cnt, events_history1
        if selected_region is None:
            print("No region selected!")
            return

        i_min, i_max, j_min, j_max = selected_region
        weights[i_min:i_max, j_min:j_max] = np.round(weights[i_min:i_max, j_min:j_max])
        im.set_data(weights)
        events_history1.append((weights.copy(), accuracy_text, accuracy_num_texts))

        plt.draw()


    def fill_callback(event):
        global selected_region, weights, events_cnt, events_history1
        if selected_region is None:
            print("No region selected!")
            return

        try:
            value = float(text_box.text)
        except ValueError:
            print("Invalid fill value!")
            return

        i_min, i_max, j_min, j_max = selected_region
        weights[i_min:i_max, j_min:j_max] = value
        im.set_data(weights)
        plt.draw()




    def run_evaluate(event):
        global weights, model1, eval_cnt, o_accuracy, o_accuracy_per_class, events_cnt, events_history1

        layer.weight = torch.nn.Parameter(torch.from_numpy(weights))
        # Initialize counters for each class
        # Initialize counters for each class
        class_correct = torch.zeros(10)
        class_total = torch.zeros(10)



        # Initialize counters for total accuracy
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data
                target = target
                output = model1(data)
                _, predicted = torch.max(output.data, 1)

                # Update total accuracy
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # Update per-class statistics
                correct_mask = (predicted == target)
                for cls in range(10):
                    cls_mask = (target == cls)
                    class_total[cls] += cls_mask.sum().item()
                    class_correct[cls] += (correct_mask & cls_mask).sum().item()

        # Calculate total accuracy
        accuracy = 100 * correct / total

        # Calculate accuracy for each digit
        accuracy_per_class = 100 * class_correct.cpu() / class_total.cpu()

        """history.append((selected_region, layer.weight.cpu().clone(),torch.sort(accuracy_per_class,descending=False)))
        with open(file, "wb") as f:
            pickle.dump(history, f)"""

        accuracy_text.set_text(f"Test Accuracy: {accuracy:.2f}%")

        if eval_cnt == 0:
            o_accuracy = accuracy
            o_accuracy_per_class = accuracy_per_class
        for cls, txt in enumerate(accuracy_num_texts):
            txt.set_text(f"{accuracy_per_class[cls]:.2f}")

        eval_cnt += 1


        plt.draw()



    def restore(event):
        global weights, model1
        if selected_region is None:
            weights = original.copy()
            layer.weight = torch.nn.Parameter(torch.from_numpy(original).to("cuda"))
        else:
            i_min, i_max, j_min, j_max = selected_region
            weights[i_min:i_max, j_min:j_max] = original[i_min:i_max, j_min:j_max].copy()
            layer.state_dict()["weight"][i_min:i_max, j_min:j_max] = torch.from_numpy(original[i_min:i_max, j_min:j_max]).to("cuda")
        im.set_data(weights)
        plt.draw()
    def restoreall(event):
        global weights, model1

        weights = original.copy()
        layer.weight = torch.nn.Parameter(torch.from_numpy(original))

        im.set_data(weights)


        plt.draw()

    def ctrlz(event):
        global weights, model1, history, events_cnt, events_history1, original, layer, im
        events_cnt += 1
        weights = layers[events_cnt].weight.cpu().detach().numpy().copy()
        original = layers[events_cnt].weight.cpu().detach().numpy().copy()
        layer = layers[events_cnt]
        im = ax.imshow(weights, cmap='viridis', aspect='auto', origin='upper')



        plt.draw()

    def invctrlz(event):
        global weights, model1, history, events_cnt, events_history1, original, layer,im

        print(events_cnt)
        events_cnt -= 1
        weights = layers[events_cnt].weight.cpu().detach().numpy().copy()
        original = layers[events_cnt].weight.cpu().detach().numpy().copy()
        layer = layers[events_cnt]
        im = ax.imshow(weights, cmap='viridis', aspect='auto', origin='upper')



        plt.draw()




    # Create widgets
    ax_round = plt.axes([0.2, 0.05, 0.05, 0.052])
    btn_round = Button(ax_round, 'Round')
    btn_round.on_clicked(round_callback)

    ax_fill = plt.axes([0.3, 0.05, 0.05, 0.052])
    btn_fill = Button(ax_fill, 'Fill')
    btn_fill.on_clicked(fill_callback)

    ax_test = plt.axes([0.1, 0.05, 0.05, 0.052])
    btn_test = Button(ax_test, 'Eval')
    btn_test.on_clicked(run_evaluate)

    ax_rest = plt.axes([0.4, 0.05, 0.05, 0.052])
    btn_rest = Button(ax_rest, 'Restore')
    btn_rest.on_clicked(restore)

    ax_resta = plt.axes([0.5, 0.05, 0.05, 0.052])
    btn_resta = Button(ax_resta, 'Restore All')
    btn_resta.on_clicked(restoreall)

    ax_l = plt.axes([0.05, 0.9, 0.05, 0.052])
    btn_l = Button(ax_l, 'Previous Layer')
    btn_l.on_clicked(invctrlz)

    ax_l1 = plt.axes([0.05, 0.8, 0.05, 0.052])
    btn_l1 = Button(ax_l1, 'Next Layer')
    btn_l1.on_clicked(ctrlz)


    """ax_z1 = plt.axes([0.0, 0.9, 0.15, 0.075])
    btn_z1 = Button(ax_z1, 'CTRLZ')
    btn_z1.on_clicked(ctrlz)

    ax_z2 = plt.axes([0.85, 0.9, 0.15, 0.075])
    btn_z2 = Button(ax_z2, 'INV CTRLZ')
    btn_z2.on_clicked(invctrlz)"""


    ax_text = plt.axes([0.6, 0.05, 0.2, 0.075])
    text_box = TextBox(ax_text, 'Fill value:', initial='0.0')

    # Hover callback
    def hover(event):
        if event.inaxes != ax:
            return

        x = event.xdata
        y = event.ydata
        j = int(round(x))
        i = int(round(y))

        if 0 <= i < weights.shape[0] and 0 <= j < weights.shape[1]:
            val = weights[i, j]
            hover_text.set_text(f'Weight [{i},{j}]: {val:.4f}')
        else:
            hover_text.set_text('')

        fig.canvas.draw_idle()


    fig.canvas.mpl_connect('motion_notify_event', hover)

    plt.show()
