import glob
import matplotlib

matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive display
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import numpy as np
from numpy import linalg
from numpy.random.mtrand import random
import scipy
from scipy import signal
from scipy.signal import hilbert
from scipy import interpolate
from scipy.stats import norm
from scipy.signal import butter, lfilter
from scipy.signal import windows
import pylops
from scipy.stats import randint
from scipy.signal import find_peaks
from scipy import stats
from scipy import ndimage, datasets
import math
# import statsmodels.api as sm
import random
from random import randint
import signal

import torch
from torch.linalg import pinv
import torch.nn as nn
import scipy as sp

from IPython.core.display import HTML
from scipy.sparse.linalg import lsqr
from torch.utils.data import TensorDataset, DataLoader
from pylops.utils.wavelets import ricker
from pylops.utils import dottest
import JsonParameters


# def some_func():
def filter_signal(signal):
    order = 3
    low_cut = 1  # 0.5
    high_cut = 10
    fs = 100
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal


def create_noise(aoi_length, signal, squeeze):
    SNR = -20  # -6
    sigma = np.max(signal) * (10 ** (SNR / 20))
    noise = sigma * np.random.randn(aoi_length)  # we can change the sigma
    noise = np.expand_dims(noise, 1)
    if (squeeze == 1):
        noise = np.squeeze(noise)
    return signal + noise

def set_seed(seed):
    """Set all random seeds to a fixed value and take out any
    randomness from cuda kernels
    Parameters
    ----------
    seed : :obj:`int`
        Seed number
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


class ContractingBlock(nn.Module):
    """Contracting block

    Single block in contracting path composed of two convolutions followed by a max pool operation.
    We allow also to optionally include a batch normalization and dropout step.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    use_dropout : :obj:`bool`, optional
        Add dropout
    use_bn : :obj:`bool`, optional
        Add batch normalization

    """

    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm1d(input_channels * 2, momentum=0.8)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


class ExpandingBlock(nn.Module):
    """Expanding block

    Single block in expanding path composed of an upsampling layer, a convolution, a concatenation of
    its output with the features at the same level in the contracting path, two additional convolutions.
    We allow also to optionally include a batch normalization and dropout step.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    use_dropout : :obj:`bool`, optional
        Add dropout
    use_bn : :obj:`bool`, optional
        Add batch normalization

    """

    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # , align_corners=False)
        self.conv1 = nn.Conv1d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(input_channels // 2, input_channels // 2, kernel_size=3, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm1d(input_channels // 2, momentum=0.8)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    """Feature Map block

    Final layer of U-Net which restores for the output channel dimensions to those of the input (or any other size)
    using a 1x1 convolution.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    output_channels : :obj:`int`
        Number of output channels

    """

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet architecture

    UNet architecture composed of a series of contracting blocks followed by expanding blocks.

    Most UNet implementations available online hard-code a certain number of levels. Here,
    the number of levels for the contracting and expanding paths can be defined by the user and the
    UNet is built in such a way that the same code can be used for any number of levels without modification.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    output_channels : :obj:`int`, optional
        Number of output channels
    hidden_channels : :obj:`int`, optional
        Number of hidden channels of first layer
    levels : :obj:`int`, optional
        Number of levels in encoding and deconding paths

    """

    def __init__(self, good_matrix_torch, input_channels=1, output_channels=1, hidden_channels=64, levels=2):
        super(UNet, self).__init__()
        self.good_matrix_torch = good_matrix_torch
        self.good_matrix_torch_hermite = self.good_matrix_torch.H
        # self.pinv_mat = pinv(good_matrix_torch)
        self.levels = levels
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract = []
        self.expand = []
        for level in range(levels):
            self.contract.append(ContractingBlock(hidden_channels * (2 ** level),
                                                  use_dropout=False, use_bn=False))
        for level in range(levels):
            self.expand.append(ExpandingBlock(hidden_channels * (2 ** (levels - level)),
                                              use_dropout=False, use_bn=False))
        self.contracts = nn.Sequential(*self.contract)
        self.expands = nn.Sequential(*self.expand)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, y):
        xenc = []
        y1 = torch.squeeze(y)
        x_hermite = torch.matmul(self.good_matrix_torch_hermite, y1.T).T

        # x_h_hilbert = hilbert(x_hermite)

        x_ = ((x_hermite - torch.min(x_hermite)) / (torch.max(x_hermite) - torch.min(x_hermite))).unsqueeze(1)
        x = self.upfeature(x_)
        xenc.append(x)
        for level in range(self.levels):
            x = self.contract[level](x)
            xenc.append(x)
        for level in range(self.levels):
            x = self.expand[level](x, xenc[self.levels - level - 1])
        xn = self.downfeature(x)
        return xn  # + x_


def train(model, criterion, optimizer, data_loader, device='gpu', plotflag=False):
    """Training step

    Perform a training step over the entire training data (1 epoch of training)

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    criterion : :obj:`torch.nn.modules.loss`
        Loss function
    optimizer : :obj:`torch.optim`
        Optimizer
    data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    device : :obj:`str`, optional
        Device
    plotflag : :obj:`bool`, optional
        Display intermediate results

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    global xpred
    model.train()
    loss = 0
    for X, y in data_loader:  # tqdm(data_loader):
        optimizer.zero_grad()
        X, y = X.unsqueeze(1), y.unsqueeze(1)
        xpred = model(y)
        ls = criterion(xpred.view(-1), X.view(-1))
        ls.backward()
        optimizer.step()
        loss += ls.item()
    loss /= len(data_loader)

    if plotflag:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.plot(((X.detach().squeeze()[:5].T[3:1158] - torch.min(X.detach().squeeze()[:5].T[3:1158])) / (
                torch.max(X.detach().squeeze()[:5].T[3:1158]) - torch.min(X.detach().squeeze()[:5].T[3:1158]))).cpu(), "k")
        ax.plot(((xpred.detach().squeeze()[:5].T[3:1158] - torch.min(xpred.detach().squeeze()[:5].T[3:1158])) / (
                torch.max(xpred.detach().squeeze()[:5].T[3:1158]) - torch.min(
            xpred.detach().squeeze()[:5].T[3:1158]))).cpu(), "r")
        ax.set_xlabel("t [pico sec]")
        plt.show()

    return loss


def evaluate(model, criterion, data_loader, device='gpu', plotflag=False):
    """Evaluation step

    Perform an evaluation step over the entire training data

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    criterion : :obj:`torch.nn.modules.loss`
        Loss function
    data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Evaluation dataloader
    device : :obj:`str`, optional
        Device
    plotflag : :obj:`bool`, optional
        Display intermediate results

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    global X
    model.train()  # not eval because https://github.com/facebookresearch/SparseConvNet/issues/166
    loss = 0
    for X, y in data_loader:  # tqdm(data_loader):
        X, y = X.unsqueeze(1), y.unsqueeze(1)
        with torch.no_grad():
            xpred = model(y)
            ls = criterion(xpred.view(-1), X.view(-1))
        loss += ls.item()
    loss /= len(data_loader)

    if plotflag:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.plot(((X.detach().squeeze()[:5].T[3:1158] - torch.min(X.detach().squeeze()[:5].T[3:1158])) / (
                torch.max(X.detach().squeeze()[:5].T[3:1158]) - torch.min(X.detach().squeeze()[:5].T[3:1158]))).cpu(), "k")
        ax.plot(((xpred.detach().squeeze()[:5].T[3:1158] - torch.min(xpred.detach().squeeze()[:5].T[3:1158])) / (
                torch.max(xpred.detach().squeeze()[:5].T[3:1158]) - torch.min(xpred.detach().squeeze()[:5].T[3:1158]))).cpu(),
                "r")
        ax.set_xlabel("t [pico sec]")
    plt.show()

    return loss


def predict(model, y, device='gpu'):
    """Prediction step

    Perform a prediction over a batch of input samples

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    y: :obj:`torch.tensor`
        Inputs
    xpred : :obj:`torch.tensor`
        Masks
    device : :obj:`str`, optional
        Device

    """
    model.train()  # not eval because https://github.com/facebookresearch/SparseConvNet/issues/166
    with torch.no_grad():
        xpred = model(y)
    return xpred


#################################################################

time, signal = np.loadtxt(JsonParameters.data_dir + '500 micron gap 100hz 200ps 3000av asops34.txt', skiprows=5,
                          unpack=True,
                          usecols=[0, 1])

aoi_length = JsonParameters.finish_aoi - JsonParameters.start_aoi
split_time = time[JsonParameters.start_aoi:JsonParameters.finish_aoi]

# pulse_8p2 after ultem, with 0 mm gap:
pulse_air_ultem = JsonParameters.data_dir + '0 micron gap 100hz 200ps 3000av asops2.txt'

time_pulse, pulse = np.loadtxt(pulse_air_ultem, skiprows=5, unpack=True, usecols=[0, 1])

##pulse_8p2 after ultem, with 0 mm gap:
pulse_length_1 = JsonParameters.finish_pulse_after_ultem - JsonParameters.start_pulse_after_ultem

filtered_pulse = filter_signal(pulse)

##pulse_8p2 after ultem, with 0 mm gap:
split_filtered_pulse_1 = filtered_pulse[JsonParameters.start_pulse_after_ultem:JsonParameters.finish_pulse_after_ultem]

good_pulse = np.zeros(JsonParameters.size_good_pulse_without_pulse_length + pulse_length_1)
split_good_time = time_pulse[0:len(good_pulse)]  # good??


window1 = scipy.signal.windows.hann(pulse_length_1)
split_filtered_pulse_1_clean = window1 * split_filtered_pulse_1
good_pulse[aoi_length - pulse_length_1:aoi_length] = split_filtered_pulse_1_clean
good_pulse = good_pulse / np.linalg.norm(good_pulse)

matrix2 = np.zeros((aoi_length, aoi_length))
for iteration in range(0, aoi_length, 1):
    matrix2[:, aoi_length - 1 - iteration] = good_pulse[iteration:iteration + aoi_length]
good_matrix = np.matrix(matrix2)
good_matrix = good_matrix / np.linalg.norm(good_matrix)
print(good_matrix)

# start deep learning
set_seed(0)

# Create data:
size = 16200
n = 0
v_avg = 137.9309 * (10 ** 6)
random_numbers = np.random.randint(10, 301, size)
vector_ultem_air_good = []
y_ultem_air_noise = []
spaces_micron = []
for space in random_numbers:
    vector_ultem_air = np.zeros(aoi_length)
    vector_ultem_air[595] = -1 / 3
    vector_ultem_air[595 + space] = 1
    vector_ultem_air_good.append(np.expand_dims(vector_ultem_air, 1))

    y_ultem_air = good_matrix * vector_ultem_air_good[n]
    y_ultem_air = np.squeeze(y_ultem_air)
    y_ultem_air_noise.append(create_noise(aoi_length, y_ultem_air, 1))
    spaces_micron.append((space / 6.25) * (10 ** (-12)) * v_avg / (10 ** (-6)))
    n = n + 1


# Train data
ntrain = 16200
X_train = np.zeros((ntrain, 1160))
y_train = np.zeros((ntrain, 1160))
y_ultem_air_noise = np.squeeze(y_ultem_air_noise)
vector_ultem_air_good = np.squeeze(vector_ultem_air_good)

for i in range(ntrain):
    y_train[i] = y_ultem_air_noise[i]
    X_train[i] = vector_ultem_air_good[i]

# Validation data
nval = 5400
X_val = np.zeros((nval, 1160))
y_val = np.zeros((nval, 1160))
y_ultem_air_noise = np.squeeze(y_ultem_air_noise)
vector_ultem_air_good = np.squeeze(vector_ultem_air_good)

for i in range(nval):
    y_val[i] = y_ultem_air_noise[i]
    X_val[i] = vector_ultem_air_good[i]

time_for_graph = np.arange(0, 200, 5 / 29)
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.plot(time_for_graph, X_train[0].T, "k", lw=3)
ax.plot(time_for_graph, y_train[0].T / np.max(y_train[0].T), "r", lw=2)
ax.set_xlabel("t [pico sec]")
ax.set_title("Model and Data", fontsize=14, fontweight="bold");

# Convert Train Set to Torch
X_train_torch = torch.from_numpy(X_train).float().to("cuda")
y_train_torch = torch.from_numpy(y_train).float().to("cuda")
train_dataset = TensorDataset(X_train_torch, y_train_torch)

# Convert Validation Set to Torch
X_val = torch.from_numpy(X_val).float().to("cuda")
y_val = torch.from_numpy(y_val).float().to("cuda")
val_dataset = TensorDataset(X_val, y_val)

# Use Pytorch's functionality to load data in batches.
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

good_matrix_torch = torch.from_numpy(good_matrix).float().to("cuda")

network = UNet(good_matrix_torch, 1, 1, hidden_channels=32)  # ,good_matrix_torch
network = network.to("cuda")
n_epochs = 100
lr = 1e-3

criterion = nn.MSELoss()
optim = torch.optim.Adam(network.parameters(), lr=lr)  # , weight_decay=1e-4)

train_loss_history = np.zeros(n_epochs)
val_loss_history = np.zeros(n_epochs)
val_loss_min = np.Inf  # track change in validation loss
consecutive_counter = 0  # Counter for consecutive iterations
for i in range(n_epochs):
    train_loss = train(network, criterion, optim,
                       train_loader, device='gpu',
                       plotflag=False)  # , good_matrix_torch
    val_loss = evaluate(network, criterion,
                         val_loader, device='gpu',
                         plotflag=True if i % 25 == 0 else False)  # , good_matrix_torch
    train_loss_history[i] = train_loss
    val_loss_history[i] = val_loss
    if i % 1 == 0:
        print(f'Epoch {i}, Training Loss {train_loss:.5f}, Test Loss {val_loss:.5f}')
        # save if validation loss decreased
        if val_loss <= val_loss_min:
            print('validation loss decreased ({:.6f} --> {:.6f}).  saving model ...'.format(
                val_loss_min,
                val_loss))
            torch.save(network.state_dict(), '//NDTGPU/Data1/Noya/SuperResolution/code/check_model_with_hermite_good_gpu.pth')
            val_loss_min = val_loss
            consecutive_counter = 0  # Reset the counter if validation loss decreases
        else:
            consecutive_counter += 1  # Increment the counter if validation loss doesn't decrease

        if consecutive_counter >= 10:
            print(
                'Validation loss has not decreased by at least 1 percent for 10 consecutive iterations. Breaking the loop.')
            break
# Let's finally display the training and test losses and a set of predictions both for training and test data
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.plot(ndimage.median_filter(train_loss_history, size=20), "k", lw=4, label='Train')
ax.plot(ndimage.median_filter(val_loss_history, size=20), "r", lw=4, label='Validation')
ax.set_title('MSE')
ax.set_xlabel('Epoch')
ax.legend()
plt.show()

network.load_state_dict(torch.load('//NDTGPU/Data1/Noya/SuperResolution/code/check_model_with_hermite_good_gpu.pth'))
network.eval()

x_train_pred = predict(network, y_train_torch[:10].unsqueeze(1), device='gpu').squeeze()
x_val_pred = predict(network, y_val[:10].unsqueeze(1), device='gpu').squeeze()

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle("Model and Prediction (Train set)", fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    ax.plot(time_for_graph[10:1150], ((X_train_torch[iax].T[10:1150] - torch.min(X_train_torch[iax].T[10:1150])) / (
            torch.max(X_train_torch[iax].T[10:1150]) - torch.min(X_train_torch[iax].T[10:1150]))).cpu(), "k", lw=2,
            label=r'$X train$')
    ax.plot(time_for_graph[10:1150], ((x_train_pred[iax].T[10:1150] - torch.min(x_train_pred[iax].T[10:1150])) / (
            torch.max(x_train_pred[iax].T[10:1150]) - torch.min(x_train_pred[iax].T[10:1150]))).cpu(), "r", lw=1,
            label=r'$X pred$')
    ax.set_xlabel("t [pico sec]")
    ax.legend()
plt.tight_layout()

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle("Model and Prediction (val set)", fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    ax.plot(time_for_graph[10:1150], ((X_val[iax].T[10:1150] - torch.min(X_val[iax].T[10:1150])) / (
            torch.max(X_val[iax].T[10:1150]) - torch.min(X_val[iax].T[10:1150]))).cpu(), "k", lw=2,
            label=r'$X val (space=%d  micron)$' % spaces_micron[iax])
    ax.plot(time_for_graph[10:1150], ((x_val_pred[iax].T[10:1150] - torch.min(x_val_pred[iax].T[10:1150])) / (
            torch.max(x_val_pred[iax].T[10:1150]) - torch.min(x_val_pred[iax].T[10:1150]))).cpu(), "r", lw=1,
            label=r'$X pred (space=%d  micron)$' % spaces_micron[iax])
    ax.set_xlabel("t [pico sec]")
    ax.legend()
plt.tight_layout()

############################ ##########################################
# now we will check with new data:
size = 5400
n = 0
v_avg = 137.9309 * (10 ** 6)
random_numbers = np.random.randint(10, 250, size)
vector_ultem_air_good_check = []
y_ultem_air_noise_check = []
spaces_micron = []
for space in random_numbers:
    vector_ultem_air = np.zeros(aoi_length)
    vector_ultem_air[595] = - 1 / 3
    vector_ultem_air[595 + space] = 1
    vector_ultem_air_good_check.append(np.expand_dims(vector_ultem_air, 1))

    y_ultem_air = good_matrix * vector_ultem_air_good_check[n]
    y_ultem_air = np.squeeze(y_ultem_air)
    y_ultem_air_noise_check.append(create_noise(aoi_length, y_ultem_air, 1))

    spaces_micron.append((space / 6.25) * (10 ** (-12)) * v_avg / (10 ** (-6)))
    n = n + 1

ntest_check = 5400
X_test_check = np.zeros((ntest_check, 1160))
y_test_check = np.zeros((ntest_check, 1160))
y_ultem_air_noise_check = np.squeeze(y_ultem_air_noise_check)
vector_ultem_air_good_check = np.squeeze(vector_ultem_air_good_check)

for i in range(ntest_check):
    y_test_check[i] = y_ultem_air_noise_check[i]
    X_test_check[i] = vector_ultem_air_good_check[i]

X_test_torch_check = torch.from_numpy(X_test_check).float().to("cuda")
y_test_torch_check = torch.from_numpy(y_test_check).float().to("cuda")
test_dataset_check = TensorDataset(X_test_torch_check, y_test_torch_check)

# Use Pytorch's functionality to load data in batches.
test_loader_check = DataLoader(test_dataset_check, batch_size=batch_size, shuffle=False)
# check new signal:
x_test_pred_check = predict(network, y_test_torch_check[:5400].unsqueeze(1), device='gpu').squeeze()
loss = criterion(x_test_pred_check, X_test_torch_check)  # [:10].unsqueeze(1))

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle("Model and Prediction A0(Test set)", fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    ax.plot(time_for_graph[10:1150],  # +6
            (((X_test_torch_check[iax].T[10:1150] - torch.min(X_test_torch_check[iax].T[10:1150])) / (
                    torch.max(X_test_torch_check[iax].T[10:1150]) - torch.min(
                X_test_torch_check[iax].T[10:1150]))) - torch.mean(
                (X_test_torch_check[iax].T[10:1150] - torch.min(X_test_torch_check[iax].T[10:1150])) / (
                        torch.max(X_test_torch_check[iax].T[10:1150]) - torch.min(
                    X_test_torch_check[iax].T[10:1150])))).cpu(),
            "k", lw=3, label=r'$X real (space=%d  micron)$' % spaces_micron[iax])  # + 0.272,
    ax.plot(time_for_graph[10:1150],
            ((x_test_pred_check[iax].T[10:1150] - torch.min(x_test_pred_check[iax].T[10:1150])) / (
                    torch.max(x_test_pred_check[iax].T[10:1150]) - torch.min(
                x_test_pred_check[iax].T[10:1150])) - torch.mean(
                (x_test_pred_check[iax].T[10:1150] - torch.min(x_test_pred_check[iax].T[10:1150])) / (
                        torch.max(x_test_pred_check[iax].T[10:1150]) - torch.min(x_test_pred_check[iax].T[10:1150])))).cpu(),
            "r",
            lw=1, label=r'$X pred (space=%d  micron)$' % spaces_micron[iax])
    ax.set_xlabel("t [pico sec]")
    ax.legend()

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle("y Test set A0", fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    ax.plot(time_for_graph[10:1150],
            ((y_test_torch_check[:10][iax].T[10:1150] - torch.min(y_test_torch_check[:10][iax].T[10:1150])) / (
                    torch.max(y_test_torch_check[:10][iax].T[10:1150]) - torch.min(
                y_test_torch_check[:10][iax].T[10:1150]))).cpu(), "b", lw=1,
            label=r'$y (space=%d  micron)$' % spaces_micron[iax])
    ax.set_xlabel("t [pico sec]")
    ax.legend()
plt.tight_layout()



fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle("Hermite A0 (Test set)", fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    x_H_check = torch.matmul(good_matrix_torch.H, y_test_torch_check[:10][iax].T).T
    ax.plot(time_for_graph[10:1150],
            ((x_H_check.T[10:1150] - torch.min(x_H_check.T[10:1150])) / (
                    torch.max(x_H_check.T[10:1150]) - torch.min(
                x_H_check.T[10:1150]))).cpu(), "g", lw=1, label=r'$matrix.H * y test (space=%d micron)$' % spaces_micron[iax])
    ax.set_xlabel("t [pico sec]")
    ax.legend()

loss_xpred_xreal = x_test_pred_check[0].T[10:1150] - X_test_torch_check[0].T[10:1150]
print(loss_xpred_xreal)
