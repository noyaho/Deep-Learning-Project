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
from scipy import interpolate
from scipy.stats import norm
from scipy.signal import butter, lfilter
from scipy.signal import windows
from scipy import ndimage, datasets
from scipy.ndimage import convolve

import pylops
from scipy.stats import randint
from scipy.signal import find_peaks
from scipy import stats
import math
import random
from random import randint
import signal

# import pyproximal
import torch
import torch.nn as nn
import scipy as sp

from IPython.core.display import HTML
from scipy.sparse.linalg import lsqr
from torch.utils.data import TensorDataset, DataLoader
from pylops.utils.wavelets import ricker
from pylops.utils import dottest

import JsonParameters
##################################

from torch.utils.data import TensorDataset, DataLoader


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


def init_weights(model):
    if type(model) == nn.Linear:
        torch.nn.init.xavier_uniform(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)
    elif type(model) == nn.Conv1d or type(model) == nn.ConvTranspose1d:
        torch.nn.init.xavier_uniform(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)


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

    # add Op???
    def __init__(self, good_matrix_torch, input_channels=1, output_channels=1, hidden_channels=64, levels=2):  # , Cop
        super(UNet, self).__init__()
        self.good_matrix_torch = good_matrix_torch
        self.good_matrix_torch_hermite = self.good_matrix_torch.H
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
        x_ = ((x_hermite - torch.min(x_hermite)) / (torch.max(x_hermite) - torch.min(x_hermite))).unsqueeze(1)
        x = self.upfeature(x_)
        xenc.append(x)
        for level in range(self.levels):
            x = self.contract[level](x)
            xenc.append(x)
        for level in range(self.levels):
            x = self.expand[level](x, xenc[self.levels - level - 1])
        xn = self.downfeature(x)
        return xn


def train(model, criterion, optimizer, data_loader, good_matrix_torch, good_matrix_torch2, mu, device='gpu', plotflag=False, param_flag=False, ls_teta=None):  # , Cop
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
    model.train()
    device = torch.device('cuda')
    model_trained = UNet(good_matrix_torch, 1, 1, hidden_channels=32)
    model_trained.load_state_dict(torch.load('//NDTGPU/Data1/Noya/SuperResolution/code/check_model_with_hermite_good_gpu.pth'))
    model_trained.to(device)
    model_trained.eval()

    if param_flag:
        filtered_state_dict = {}
        for name in model.state_dict():
            if name in model_trained.state_dict():
                filtered_state_dict[name] = model_trained.state_dict()[name]

        model.load_state_dict(filtered_state_dict, strict=False)

    loss = 0
    # count_iter = 0
    for X, y in data_loader:  # tqdm(data_loader):
        # count_iter = count_iter + 1
        # print(count_iter)
        optimizer.zero_grad()
        X, y = X.unsqueeze(1), y.unsqueeze(1)
        xmodel = model(y)
        xmodel2 = torch.squeeze(xmodel)

        xpred_mat = torch.matmul(good_matrix_torch2, xmodel2.T).T
        xpred = xpred_mat.unsqueeze(1)
        ls_teta = sum((p - k).pow(2.0).sum()
                  for p, k in zip(model.parameters(), model_trained.parameters()))
        ls = criterion(xpred, y) + mu * ls_teta  # view(-1)
        ls.backward()
        optimizer.step()
        loss += ls.item()
    loss /= len(data_loader)

    return loss


def evaluate(model, criterion, data_loader, good_matrix_torch, good_matrix_torch2, mu, device='gpu', plotflag=False):  # , Cop
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
    model.train()
    device = torch.device('cuda')
    model_trained = UNet(good_matrix_torch, 1, 1, hidden_channels=32)
    model_trained.load_state_dict(torch.load('//NDTGPU/Data1/Noya/SuperResolution/code/check_model_with_hermite_good_gpu.pth'))
    model_trained.to(device)
    model_trained.eval()

    loss = 0
    count_iter = 0
    for X, y in data_loader:  # tqdm(data_loader):
        count_iter = count_iter + 1
        print(count_iter)
        X, y = X.unsqueeze(1), y.unsqueeze(1)
        with torch.no_grad():
            xmodel = model(y)
            xmodel2 = torch.squeeze(xmodel)
            # xpred = xmodel * pylops.TorchOperator(Cop_air)
            xpred_mat = torch.matmul(good_matrix_torch2, xmodel2.T).T ##########
            xpred = xpred_mat.unsqueeze(1)
            # xpred = xmodel * good_matrix_torch
            ls_teta = sum((p - k).pow(2.0).sum()
                          for p, k in zip(model.parameters(), model_trained.parameters()))
            ls = criterion(xpred, y) + mu * ls_teta  # .view(-1)
        loss += ls.item()
    loss /= len(data_loader)

    return loss


def predict(model, y, device='gpu'):  # device='cpu'
    """Prediction step

    Perform a prediction over a batch of input samples

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    X : :obj:`torch.tensor`
        Inputs
    X : :obj:`torch.tensor`
        Masks
    device : :obj:`str`, optional
        Device

    """
    model.train()  # not eval because https://github.com/facebookresearch/SparseConvNet/issues/166
    with torch.no_grad():
        xpred = model(y)
    return xpred

########################################################
def find_pulse_start_finish(signal):
    # Find indices where signal is non-zero
    non_zero_indices = torch.nonzero(signal).squeeze()
    # Find the first and last non-zero index
    start_index = non_zero_indices[0]
    finish_index = non_zero_indices[-1]
    return start_index, finish_index

def attenuate_pulse(good_pulse):
    f_sample = 10  # sampling rate [Thz]
    dt = 1 / f_sample  # [picoseconds]
    N = 1000  # signal length
    t_arr = torch.linspace(0, N * dt, N)
    f0 = 1.0  # central frequency of the simulate initial pulse_8p2 [Thz]
    phi = torch.pi / 2  # phase [rad]
    tau = 50  # posintion of pulse_8p2 in time [picoseconds]
    sigma = 0.5  # width of signal [pico second]

    # Attenuation constants  [1/cm]
    # taken from reference "Pulsed THz imaging for thickness characterization of plastic sheets", NDT&EInternational,2020
    alpha0 = 0.388
    alpha1 = 13.15       # second term of attenuation
    alpha_p = 1.777

    length = 1.5  # distance of travel [cm]

    # simulate the initial pulse_8p2 as a Gabor function
    initial_pulse = torch.from_numpy(good_pulse[500:1500]).float()

    # calculate the fft of the initial pulse_8p2
    initial_pulse_fft = torch.fft.rfft(initial_pulse)
    f_arr = torch.fft.rfftfreq(N, dt)

    # Apply the attenuation on the initial pulse_8p2 (in the frequency domain)
    alpha_arr = alpha0 + alpha1 * torch.pow(torch.abs(f_arr), alpha_p)
    attenuation_arr = torch.exp(-alpha_arr * length)
    atteunated_pulse_fft = attenuation_arr * initial_pulse_fft

    # Return  back to the time domain
    atteunated_pulse = torch.real(torch.fft.irfft(atteunated_pulse_fft))
    atteunated_pulse[np.abs(atteunated_pulse) < 0.004] = 0

    # ploting
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t_arr, initial_pulse / np.linalg.norm(initial_pulse), label='Initial pulse_8p2')
    ax[0].plot(t_arr, atteunated_pulse / np.linalg.norm(atteunated_pulse), label='Attenuated pulse_8p2')
    ax[0].set_xlabel('time[picoseconds]')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(f_arr, torch.abs(initial_pulse_fft), label='Initial Spectrum')
    ax[1].set_xlabel('frequency [Thz]')
    ax[1].set_ylabel('Amplitude')
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t_arr, atteunated_pulse, label='Attenuated pulse_8p2')
    ax[0].set_xlabel('time[picoseconds]')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(f_arr, torch.abs(atteunated_pulse_fft), label='Attenuated Spectrum')
    ax[1].set_xlabel('frequency [Thz]')
    ax[1].set_ylabel('Amplitude')

    find_pulse_start_finish(initial_pulse)
    atteunated_pulse_start_finish = find_pulse_start_finish(atteunated_pulse)
    start_attenuated_pulse = atteunated_pulse_start_finish[0]
    finish_attenuated_pulse = atteunated_pulse_start_finish[1]
    return atteunated_pulse, start_attenuated_pulse, finish_attenuated_pulse

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
    set_seed(0)
    SNR = -20  # -6
    sigma = np.max(signal) * (10 ** (SNR / 20))
    noise = sigma * np.random.randn(aoi_length)  # we can change the sigma
    noise = np.expand_dims(noise, 1)
    if (squeeze == 1):
        noise = np.squeeze(noise)
    return signal + noise

#################################################################

time, signal = np.loadtxt(JsonParameters.data_dir + '500 micron gap 100hz 200ps 3000av asops34.txt', skiprows=5,
                          unpack=True,
                          usecols=[0, 1])

aoi_length = JsonParameters.finish_aoi - JsonParameters.start_aoi
split_time = time[JsonParameters.start_aoi:JsonParameters.finish_aoi]

# pulse_8p2 from palbam without ultem:
pulse_only_palbam_pulse_filename = JsonParameters.data_dir + 'refernce reflection palbam at focus 100hz 200ps 3000av no ultam asops51.txt'
time_pulse_only_palbam, pulse_only_palbam = np.loadtxt(pulse_only_palbam_pulse_filename, skiprows=5, unpack=True,
                                                       usecols=[0, 1])

# pulse_8p2 after ultem, with 0 mm gap:
pulse_air_ultem = JsonParameters.data_dir + '0 micron gap 100hz 200ps 3000av asops2.txt'

# pulse_8p2 after palbam that after ultem and 5000 micron gap:
pulse_air_palbam = JsonParameters.data_dir + '5000 micron gap 100hz 200ps 3000av asops50.txt'

pulse_filename = pulse_air_ultem
time_pulse, pulse = np.loadtxt(pulse_filename, skiprows=5, unpack=True, usecols=[0, 1])

##pulse_8p2 after ultem, with 0 mm gap:
pulse_length_1 = JsonParameters.finish_pulse_after_ultem - JsonParameters.start_pulse_after_ultem

filtered_pulse = filter_signal(pulse)

##pulse_8p2 after ultem, with 0 mm gap:
split_filtered_pulse_1 = filtered_pulse[JsonParameters.start_pulse_after_ultem:JsonParameters.finish_pulse_after_ultem]

good_pulse = np.zeros(JsonParameters.size_good_pulse_without_pulse_length + pulse_length_1)  #+1589)
split_good_time = time_pulse[0:len(good_pulse)]   #good??

window1 = scipy.signal.windows.hann(pulse_length_1)
split_filtered_pulse_1_clean = window1 * split_filtered_pulse_1
good_pulse[aoi_length - pulse_length_1:aoi_length] = split_filtered_pulse_1_clean
good_pulse = good_pulse / np.linalg.norm(good_pulse)

plt.figure()
plt.plot(split_good_time, good_pulse)
plt.title('pulse_8p2 after ultem - with 0 mm gap')
plt.xlabel('time [psec]')
# plt.ylabel()
plt.grid()
plt.show()


matrix2 = np.zeros((aoi_length, aoi_length))
for iteration in range(0, aoi_length, 1):
    matrix2[:, aoi_length - 1 - iteration] = good_pulse[iteration:iteration + aoi_length]
good_matrix = np.matrix(matrix2)
good_matrix = good_matrix / np.linalg.norm(good_matrix)
print(good_matrix)

good_pulse2 = np.zeros(JsonParameters.size_good_pulse_without_pulse_length + pulse_length_1)
attenuated_good_pulse, start_attenuated_good_pulse, finish_attenuated_good_pulse = attenuate_pulse(good_pulse)
pulse_length_2 = finish_attenuated_good_pulse - start_attenuated_good_pulse
# window2 = scipy.signal.windows.hann(pulse_length_2)
window2 = torch.hann_window(pulse_length_2)
split_attenuated_good_pulse = attenuated_good_pulse[start_attenuated_good_pulse:finish_attenuated_good_pulse]
split_attenuated_good_pulse_clean = window2 * split_attenuated_good_pulse
good_pulse2[aoi_length - (pulse_length_2):aoi_length] = split_attenuated_good_pulse_clean
good_pulse2 = good_pulse2 / np.linalg.norm(good_pulse2)

plt.figure()
plt.plot(split_good_time, good_pulse2)
plt.title('pulse_8p2 after ultem - with 0 mm gap - attenuated pulse_8p2')
plt.xlabel('time [psec]')
# plt.ylabel()
plt.grid()
# plt.show()


matrix22 = np.zeros((aoi_length, aoi_length))
for iteration in range(0, aoi_length, 1):
    matrix22[:, aoi_length - 1 - iteration] = good_pulse2[iteration:iteration + aoi_length]
good_matrix2 = np.matrix(matrix22)
good_matrix2 = good_matrix2 / np.linalg.norm(good_matrix2)
print(good_matrix2)

plt.figure()
plt.plot(split_good_time, good_pulse, 'b', label='good_pulse 1')
plt.plot(split_good_time, good_pulse2, 'g', label='good_pulse 2-attenuated pulse_8p2')
plt.legend()
plt.xlabel('time [psec]')
plt.grid()


# vector = np.zeros(aoi_length)

time_for_graph = np.arange(0, 200, 5 / 29)
# start deep learning
set_seed(0)

size = 16200  # 10
n = 0
v_avg = 137.9309 * (10 ** 6)
random_numbers = np.random.randint(10, 301, size)
vector_ultem_air_good = []
y_ultem_air_noise = []
space_choose = 62
spaces_micron = []
for i in range(size):
    vector_ultem_air = np.zeros(aoi_length)
    vector_ultem_air[595] = - 1 / 3
    vector_ultem_air[595 + space_choose] = 1
    vector_ultem_air_good.append(np.expand_dims(vector_ultem_air, 1))

    y_ultem_air = good_matrix2 * vector_ultem_air_good[n]
    y_ultem_air = np.squeeze(y_ultem_air)
    y_ultem_air_noise.append(create_noise(aoi_length, y_ultem_air, 1))
    n = n + 1

spaces_micron = int((space_choose / 6.25) * (10 ** (-12)) * v_avg / (10 ** (-6)))


ntrain_check = 16200
X_train_A1 = np.zeros((ntrain_check, 1160))
y_train_A1 = np.zeros((ntrain_check, 1160))
y_ultem_air_noise = np.squeeze(y_ultem_air_noise)
vector_ultem_air_good = np.squeeze(vector_ultem_air_good)

for i in range(ntrain_check):
    y_train_A1[i] = y_ultem_air_noise[i]
    X_train_A1[i] = vector_ultem_air_good[i]

X_train_A1_torch = torch.from_numpy(X_train_A1).float().to("cuda")
y_train_A1_torch = torch.from_numpy(y_train_A1).float().to("cuda")
train_dataset_A1 = TensorDataset(X_train_A1_torch, y_train_A1_torch)


nval_A1 = 5400
X_val_A1 = np.zeros((nval_A1, 1160))
y_val_A1 = np.zeros((nval_A1, 1160))
y_ultem_air_noise = np.squeeze(y_ultem_air_noise)
vector_ultem_air_good = np.squeeze(vector_ultem_air_good)

for i in range(nval_A1):
    y_val_A1[i] = y_ultem_air_noise[i]
    X_val_A1[i] = vector_ultem_air_good[i]

X_val_A1_torch = torch.from_numpy(X_val_A1).float().to("cuda")
y_val_A1_torch = torch.from_numpy(y_val_A1).float().to("cuda")
val_dataset_A1 = TensorDataset(X_val_A1_torch, y_val_A1_torch)

# Use Pytorch's functionality to load data in batches.
batch_size = 64

train_loader_A1 = DataLoader(train_dataset_A1, batch_size=batch_size, shuffle=False)
val_loader_A1 = DataLoader(val_dataset_A1, batch_size=batch_size, shuffle=False)


good_matrix_torch = torch.from_numpy(good_matrix).float().to("cuda")
good_matrix_torch2 = torch.from_numpy(good_matrix2).float().to("cuda")

device = torch.device('cuda')
network = UNet(good_matrix_torch, 1, 1, hidden_channels=32)  # ,good_matrix_torch
network.load_state_dict(torch.load('//NDTGPU/Data1/Noya/SuperResolution/code/check_model_with_hermite_good.pth'))
network.to(device)
network.eval()

criterion = nn.MSELoss()
lr = 1e-3
optim = torch.optim.Adam(network.parameters(), lr=lr)  # , weight_decay=1e-4)


# check new signal:
x_pred_A1_on_A0 = predict(network, y_val_A1_torch[:10].unsqueeze(1), device='gpu').squeeze()

# the batch loss
loss = criterion(x_pred_A1_on_A0, X_val_A1_torch[:10])  ##

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle("Model A1 and Prediction - on train A0 (Test set)", fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    ax.plot(time_for_graph[10:1150],
            (((X_val_A1_torch[iax].T[10:1150] - torch.min(X_val_A1_torch[iax].T[10:1150])) / (
                    torch.max(X_val_A1_torch[iax].T[10:1150]) - torch.min(
                X_val_A1_torch[iax].T[10:1150]))) - torch.mean((X_val_A1_torch[iax].T[10:1150] - torch.min(X_val_A1_torch[iax].T[10:1150])) / (
                    torch.max(X_val_A1_torch[iax].T[10:1150]) - torch.min(
                X_val_A1_torch[iax].T[10:1150])))).cpu().numpy(),
            "k", lw=3, label=r'$X real (space=%d  micron)$' % spaces_micron)  # + 0.272,
    ax.plot(time_for_graph[10:1150],
            ((x_pred_A1_on_A0[iax].T[10:1150] - torch.min(x_pred_A1_on_A0[iax].T[10:1150])) / (
                    torch.max(x_pred_A1_on_A0[iax].T[10:1150]) - torch.min(
                x_pred_A1_on_A0[iax].T[10:1150])) - torch.mean(
                (x_pred_A1_on_A0[iax].T[10:1150] - torch.min(x_pred_A1_on_A0[iax].T[10:1150])) / (
                        torch.max(x_pred_A1_on_A0[iax].T[10:1150]) - torch.min(x_pred_A1_on_A0[iax].T[10:1150])))).cpu().numpy(), "r",
            lw=1, label=r'$X pred (space=%d  micron)$' % spaces_micron)
    ax.set_xlabel("t [pico sec]")
    ax.legend()

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle("y A1 - on train A0", fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    ax.plot(time_for_graph[10:1150],
            ((y_val_A1_torch[:10][iax].T[10:1150] - torch.min(y_val_A1_torch[:10][iax].T[10:1150])) / (
                    torch.max(y_val_A1_torch[:10][iax].T[10:1150]) - torch.min(
                y_val_A1_torch[:10][iax].T[10:1150]))).cpu().numpy(), "b", lw=1,
            label=r'$y (space=%d  micron)$' % spaces_micron)
    ax.set_xlabel("t [pico sec]")
    ax.legend()
plt.tight_layout()

device = torch.device('cuda')
network = UNet(good_matrix_torch2, 1, 1, hidden_channels=32)  # Cop_air,
network.to(device)

n_epochs = 100
lr = 1e-3
mu = 1e-5
criterion = nn.MSELoss()
optim = torch.optim.Adam(network.parameters(), lr=lr) #, weight_decay=1e-4)

train_loss_history_A1 = np.zeros(n_epochs)
val_loss_history_A1 = np.zeros(n_epochs)
val_loss_min = np.Inf  # track change in validation loss
consecutive_counter = 0  # Counter for consecutive iterations
for i in range(n_epochs):
    train_A1_loss = train(network, criterion, optim,
                          train_loader_A1, good_matrix_torch, good_matrix_torch2, mu, device='gpu',
                          plotflag=False, param_flag=True if i == 0 else False)
    val_A1_loss = evaluate(network, criterion,
                           val_loader_A1, good_matrix_torch, good_matrix_torch2, mu, device='gpu',
                           plotflag=True if i % 25 == 0 else False)
    train_loss_history_A1[i] = train_A1_loss
    val_loss_history_A1[i] = val_A1_loss
    if i % 1 == 0:
        print(f'Epoch {i}, Training Loss {train_A1_loss:.5f}, Test Loss {val_A1_loss:.5f}')
        # save if validation loss decreased
        if val_A1_loss <= val_loss_min:
            print('validation loss decreased ({:.6f} --> {:.6f}).  saving model ...'.format(
                val_loss_min,
                val_A1_loss))
            torch.save(network.state_dict(), '//NDTGPU/Data1/Noya/SuperResolution/code/check_step2_model_A1_with_hermite_good_mu={}.pth'.format(mu))
            val_loss_min = val_A1_loss
            consecutive_counter = 0  # Reset the counter if validation loss decreases
        else:
            consecutive_counter += 1  # Increment the counter if validation loss doesn't decrease

        if consecutive_counter >= 10:
            print(
                'Validation loss has not decreased by at least 1 percent for 10 consecutive iterations. Breaking the loop.')
            break


# Let's finally display the training and test losses and a set of predictions both for training and test data
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.plot(ndimage.median_filter(train_loss_history_A1, size=20), "k", lw=4, label='Train')
ax.plot(ndimage.median_filter(val_loss_history_A1, size=20), "r", lw=4, label='Validation')
ax.axhline(y=min(val_loss_history_A1), color='g', linestyle='--', linewidth=2, label=r'$min loss=%f $' % min(val_loss_history_A1))
ax.set_title(r'$MSE A1, mu=%f $' % mu)
ax.set_xlabel('Epoch')
ax.legend()
plt.show()

device = torch.device('cuda')
network.load_state_dict(torch.load('//NDTGPU/Data1/Noya/SuperResolution/code/check_step2_model_A1_with_hermite_good_mu={}.pth'.format(mu)))
network.to(device)
network.eval()

x_ttt_pred_A1 = predict(network, y_val_A1_torch[:10].unsqueeze(1), device='gpu').squeeze()

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(14, 6))
fig.suptitle(r"$xreal & xpred, mu=%f (Test set)$" % mu, fontsize=14, fontweight="bold")
axs = axs.ravel()
for iax, ax in enumerate(axs):
    ax.plot(time_for_graph[10:1150],
            (((X_val_A1_torch[iax].T[10:1150] - torch.min(X_val_A1_torch[iax].T[10:1150])) / (
                    torch.max(X_val_A1_torch[iax].T[10:1150]) - torch.min(
                X_val_A1_torch[iax].T[10:1150]))) - torch.mean((X_val_A1_torch[iax].T[10:1150] - torch.min(X_val_A1_torch[iax].T[10:1150])) / (
                    torch.max(X_val_A1_torch[iax].T[10:1150]) - torch.min(
                X_val_A1_torch[iax].T[10:1150])))).cpu().numpy(),
            "k", lw=3, label=r'$X real (space=%d  micron)$' % spaces_micron)  # + 0.272,
    ax.plot(time_for_graph[10:1150],
            ((x_ttt_pred_A1[iax].T[10:1150] - torch.min(x_ttt_pred_A1[iax].T[10:1150])) / (
                    torch.max(x_ttt_pred_A1[iax].T[10:1150]) - torch.min(
                x_ttt_pred_A1[iax].T[10:1150])) - torch.mean(
                (x_ttt_pred_A1[iax].T[10:1150] - torch.min(x_ttt_pred_A1[iax].T[10:1150])) / (
                        torch.max(x_ttt_pred_A1[iax].T[10:1150]) - torch.min(x_ttt_pred_A1[iax].T[10:1150])))).cpu().numpy(),
            "r",
            lw=1, label=r'$X pred (space=%d  micron)$' % spaces_micron)
    ax.set_xlabel("t [pico sec]")
    ax.legend()



sigma = 2
size = 50
x = np.linspace(-size // 2, size // 2, size)
kernel = np.exp(-x**2 / (2 * sigma**2))
kernel /= np.sum(kernel)

# Convert the signal and kernel to numpy arrays
# signal_np = x_ttt_pred_optim_A.cpu()
kernel_np = kernel.reshape(1, -1)

# Perform convolution
convolved_signal_result = convolve(x_ttt_pred_A1.cpu(), kernel_np)
convolved_signal_train = convolve(X_val_A1_torch.cpu(), kernel_np)

convolved_signal_result = convolved_signal_result / np.linalg.norm(convolved_signal_result)
convolved_signal_train = convolved_signal_train / np.linalg.norm(convolved_signal_train)

# Convert the convolved signal back to a Torch tensor
convolved_signal_result = torch.from_numpy(convolved_signal_result)

plt.figure(); plt.plot(convolved_signal_result[0].T[10:1150].cpu() / np.linalg.norm(convolved_signal_result[0].T[10:1150].cpu()))
plt.plot(X_val_A1_torch[0].T[10:1150].cpu())  # / np.linalg.norm(X_val_A1_torch[0].T[10:1150]))
plt.plot(x_ttt_pred_A1[0].T[10:1150].cpu())  # / np.linalg.norm(x_ttt_pred_A1[0].T[10:1150].cpu()))
plt.plot(convolved_signal_train[0].T[10:1150] / np.linalg.norm(convolved_signal_train[0].T[10:1150]))

loss_xpred_xreal = criterion(torch.from_numpy(convolved_signal_train[0].T[10:1150]).float(), convolved_signal_result[0].T[10:1150])
# loss_xpred_xreal = convolved_signal_result[0].T[10:1150].cpu() - convolved_signal_train[0].T[10:1150]
# loss_xpred_xreal = x_ttt_pred_A1[0].T[10:1150] - X_train_A1_torch[0].T[10:1150]
# max_loss_xpred_xreal = torch.max(loss_xpred_xreal)
print(loss_xpred_xreal)
