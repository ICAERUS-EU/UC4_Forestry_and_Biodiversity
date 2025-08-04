from torch import nn
import torch
import numpy as np
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import math
import sys
from cubes import gather_cubes, bbox_cutter, min_max_scaler
from spectral.cube import Cube
from torch.utils.data import Dataset, DataLoader
import copy
from baselib.base import LibraryBase
from parameters import params
import argparse

### Functions and models -----------------------------------------------

class Savgol(LibraryBase):
    def __init__(self, window_length, polyorder, deriv=0, delta=1.0, axis=-1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.axis = axis

    def forward(self, X, y=None):
        return savgol_filter(X, self.window_length, self.polyorder, self.deriv, self.delta, self.axis)


# K-means clusters initialisation
def kmeans(model, dataloader, device):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        _, _, outputs = model(inputs)  # model output: Reconstruction, clustering out, Latent space.
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 100000: break

    # Perform K-means
    km.fit_predict(output_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(device))
    # torch.cuda.empty_cache()


# K-means clusters initialisation
def kmeans_only(model, dataloader, device, n_clusters=None):
    km = KMeans(n_clusters=n_clusters, n_init=10)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        _, _, outputs = model(inputs)  # model output: Reconstruction, clustering out, Latent space.
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if output_array.shape[0] > 100000: break

    # Perform K-means
    km.fit(output_array)
    return km


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, device):
    output_array = None
    model.eval()
    for data in dataloader:
        inputs, idxs = data
        # inputs = inputs.to(device)
        _, outputs, _ = model(inputs)  # model output: Reconstruction, clustering out, Latent space.
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    return output_array, preds

def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist


def out_size_calc(in_size, kernel, stride=1, padding=0, dilation=1):
    assert len(in_size) == len(kernel)
    assert len(in_size) > 1
    new_size = [0] * len(in_size)
    for i in range(len(in_size)):
        new_size[i] = math.floor((in_size[i] + 2 * padding - dilation * (kernel[i] - 1) -1 )/ stride + 1)
    return new_size

def out_size_calc_transpose(in_size, kernel, stride=1, padding=0, dilation=1, out_padding=0):
    assert len(in_size) == len(kernel)
    assert len(in_size) > 1
    new_size = [0] * len(in_size)
    for i in range(len(in_size)):
        new_size[i] = (in_size[i] - 1) * stride - 2 * padding + dilation * (kernel[i] - 1) + out_padding + 1
    return new_size


class HyperData(Dataset):
    def __init__(self, data_cube: Cube, min1: int = 0, max1: int = None, device: str = 'cpu', input_shape=[5, 5, 224], smoothing=None):
        self.min1 = min1
        self.max1 = max1
        self.orig_shape = data_cube.shape
        self.data = min_max_scaler(data_cube.data, self.min1, self.max1)
        self.data.astype("float32")
        self.smoothing = smoothing
        if self.smoothing is not None:
            self.data = self.smoothing(self.data)
        # trim image to have multiple of 5 x and y coords
        self.input_shape = input_shape
        self.input_x = input_shape[0]
        self.input_y = input_shape[1]
        self.xsize = self.orig_shape[0] // self.input_x
        self.ysize = self.orig_shape[1] // self.input_y
        self.device = device
 
    def __len__(self):
        return self.xsize * self.ysize

    def __getitem__(self, idx):
        # part out shape has to be [1 x 5 x 5 x bands]
        posy = idx // self.xsize
        posx = idx - (self.xsize * posy)
        part = self.data[posx:(posx+self.input_x), posy:(posy+self.input_y), :]
        if self.device == 'cpu':
            part = torch.tensor(part, dtype=torch.float32).unsqueeze(0)
        else:
            part = torch.tensor(part, dtype=torch.float32).unsqueeze(0).to(self.device)
        return part, idx


class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)

# Based on https://github.com/michaal94/torch_DCEC
class CAE(nn.Module):
    def __init__(self, input_shape=[5, 5, 224], kernel=[3, 3, 5], num_clusters=10, filters=32, leaky=True, neg_slope=0.01, activations=False, bias=True, dropout=0.5, latent_space=50):
        super(CAE, self).__init__()
        self.input_shape = input_shape
        self.num_clusters = num_clusters
        self.filters = filters
        self.leaky = leaky
        self.neg_slope = neg_slope
        self.bias = bias
        self.activations = activations
        self.kernel = kernel
        self.dropout = dropout
        self.latent_space = latent_space

        self.conv1 = nn.Conv3d(1, filters, self.kernel, bias=bias)
        self.out_shape1 = out_size_calc(self.input_shape, self.kernel)
        self.drop1  = nn.Dropout(self.dropout)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = nn.Conv3d(filters, filters, self.kernel, bias=bias)
        self.out_shape2 = out_size_calc(self.out_shape1, self.kernel)
        self.act2 = nn.ReLU()

        self.embedding = nn.Linear(in_features=self.out_shape2[0] * self.out_shape2[1] * self.out_shape2[2] * filters, out_features=self.latent_space)
        self.deembedding = nn.Linear(out_features=self.out_shape2[0] * self.out_shape2[1] * self.out_shape2[2] * filters, in_features=self.latent_space)
        # latent to cluster space
        if self.num_clusters != self.latent_space:
            self.adapter = nn.Linear(in_features=self.latent_space, out_features=self.num_clusters)
        else:
            self.adapter = nn.Identity()

        self.conv3 = nn.ConvTranspose3d(filters, filters, self.kernel, bias=bias)
        self.out_shape3 = out_size_calc_transpose(self.out_shape2, self.kernel)
        self.drop3 = nn.Dropout(self.dropout)
        self.act3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose3d(filters, 1, self.kernel, bias=bias)
        self.out_shape4 = out_size_calc_transpose(self.out_shape3, self.kernel)

        self.clustering = ClusterlingLayer(self.num_clusters, self.num_clusters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        # x = self.drop1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = x.view(-1, self.out_shape2[0] * self.out_shape2[1] * self.out_shape2[2] * self.filters)
        x = self.embedding(x)
        latent = x  # batch, 50 <- latent_space
        latent = self.adapter(x)
        clusters = self.clustering(latent)
        x = self.deembedding(x)
        x = x.view(-1, self.filters, self.out_shape2[0], self.out_shape2[1], self.out_shape2[2])
        x = self.conv3(x)
        x = self.act3(x)
        # x = self.drop3(x)
        x = self.conv4(x)
        return x, clusters, latent


### Main body -------------------------------------------------------

def result(model, input_shape, data_cube: Cube, device, min1: int = 0, max1: int = None, smoothing=None):
    if max1 is None:
        max1 = data_cube.data.max()
    size = data_cube.shape
    data = min_max_scaler(data_cube.data, min1, max1)
    np.astype(data, "float32")
    if smoothing is not None:
        data = smoothing(data)
    results = np.zeros((size[0], size[1]), dtype="uint8")
    size_x = input_shape[0]
    size_y = input_shape[1]
    xsize = size[0] // size_x
    ysize = size[1] // size_y
    for x in range(xsize):
        for y in range(ysize):
            _, _, res = model(torch.tensor(data[np.newaxis, x*size_x:(x*size_x+size_x), y*size_y:(y*size_y+size_y), :], dtype=torch.float32).unsqueeze(0).to(device))
            results[x*size_x:(x*size_x+size_x), y*size_y:(y*size_y+size_y)] = np.argmax(res.cpu().detach().numpy())
    return results


def train(params, loader, device):
    model = CAE(kernel=params["kernel"], latent_space=params["latent_space"], num_clusters=params["num_clusters"],
                filters=params["filters"], input_shape=params["inshape"]).to(device)
    loss_fn = nn.L1Loss(size_average=True)
    loss_clust = nn.KLDivLoss(size_average=False)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params["LR"], weight_decay=params["weight_decay"])

    # init weights
    kmeans(model, copy.deepcopy(loader), device)
    
    # initial distribution
    print("Calc distribution")
    output_distribution, preds = calculate_predictions(model, loader, device)
    target_distribution = target(output_distribution)
    
    print("Running model")
    rec_loss = 0
    clust_loss = 0
    losses = []
    for epoch in range(params["n_epochs"]):
        # Keep the batch number for inter-phase statistics
        batch_num = 1
        rec_loss = 0
        clust_loss = 0
        for X, idx in loader:
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                output_distribution, preds = calculate_predictions(model, loader, device)
                target_distribution = target(output_distribution)
    
            tar_dist = target_distribution[((batch_num - 1) * batch_size):(batch_num*batch_size), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            # forward pass
            y_pred, clusters, latent = model(X)
            loss1 = loss_fn(y_pred, X)
            loss2 = params["gamma"] * loss_clust(torch.log(clusters), tar_dist) / batch_size
            loss = loss1 + loss2
            rec_loss += loss1
            clust_loss += loss2
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            batch_num = batch_num + 1
        losses.append(loss.data.cpu().numpy())
        print(f"Epoch {epoch}, rec loss: {rec_loss}, clust loss: {clust_loss}")
    return model, losses


def Main(vars):
    # parse parameters
    main_params = copy.deepcopy(params)
    assert vars['device'] in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], "Please provide a valid device. Options: cpu, cuda, cuda:#"

    data_cube = Cube(vars["cube"])

    if vars['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if main_params["smoothing"]:
        sf = Savgol(15, 3)
        data = HyperData(data_cube, device=device, input_shape=main_params["inshape"], smoothing=sf)
    else:
        data = HyperData(data_cube, device=device, input_shape=main_params["inshape"], smoothing=None)
        
    loader  = DataLoader(data, batch_size=main_params["batch_size"], shuffle=False)
    print("loader length: ", len(loader))

    model, losses = train(main_params, loader, device)

    del loader
    del data

    print("Generate results")
    if main_params["smoothing"]:
        res = result(model, input_shape=main_params["inshape"], data_cube=data_cube, device=device, smoothing=sf)
    else:
        res = result(model, input_shape=main_params["inshape"], data_cube=data_cube, device=device, smoothing=None)
    if vars['out'] is None:
        np.save(vars['cube'].replace(".dat", ".npy"), res)
    else:
        np.save(vars['out'], res)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on. Default: cpu. Options: cpu, cuda.')
    parser.add_argument('--cube', type=str, help='Path to hyperspectral cube (.dat, or other ENVI format) file.', required=True)
    parser.add_argument('--out', type=str, help='Path to result output location. Default: None, use cube path', required=True, default=None)
    
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    opt = vars(opt)
    Main(opt)
