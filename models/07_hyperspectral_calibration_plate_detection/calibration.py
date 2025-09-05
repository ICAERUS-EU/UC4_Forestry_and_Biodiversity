import argparse
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from spectral.cube import Cube
from spectral.specim import MinMaxScaler, SpecimFullImageWriter
from sklearn.preprocessing import normalize as vector_norm

MODEL_PARAMETERS = {"base_model": {"weights": "../weights/hyperspectral_calibration_plate_detection_v2.pth", "in_shape": 224, "out_shape": 7}}


def out_size_calc(in_size, kernel, stride=1, padding=0, dilation=1):
    return math.floor((in_size + 2 * padding - dilation * (kernel - 1) -1) / stride + 1)


def min_max_scaler(Y, miny=None, maxy=None):
    if miny is None:
        miny = Y.min()
        print("Min: ", miny)
    if maxy is None:
        maxy = Y.max()
        print("Max: ", maxy)
    return (Y - miny) / (maxy - miny)


class CalibrationModel(nn.Module):
    def __init__(self, in_shape, out_shape, dropout=0.15, stride=1):
        super().__init__()
        self.in_shape = in_shape  # num of bands 224
        self.out_shape = out_shape  # num of classes 3 + 1 background
        self.dropout = dropout
        self.stride = stride
        self.kernel = 50

        self.conv1 = nn.Conv1d(1, 64, self.kernel, stride=self.stride)
        self.conv1_bn = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.conv_out1 = out_size_calc(self.in_shape, self.kernel, stride=self.stride)

        self.conv2 = nn.Conv1d(64, 92, self.kernel)
        self.conv2_bn = nn.BatchNorm1d(92)
        self.act2 = nn.ReLU()
        self.conv_out2 = out_size_calc(self.conv_out1, self.kernel, stride=self.stride)

        self.conv3 = nn.Conv1d(92, 128, self.kernel)
        self.act3 = nn.ReLU()
        self.conv_out3 = out_size_calc(self.conv_out2, self.kernel, stride=self.stride)

        self.fc1 = nn.Linear(self.conv_out3 * 128, 256)
        self.act4 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(256, self.out_shape)
        self.actf = nn.Identity()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = self.act1(self.conv1_bn(self.conv1(x)))
        x = self.act2(self.conv2_bn(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = x.view(-1, self.conv_out3 * 128)  # 128 - size of last conv layer channels
        x = self.act4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.actf(x)
        return x


class CalibrationPlateClassifier:

    def __init__(self, model: nn.Module, parameters: dict, device='cpu', scaler="minmax", batch_size=20000, **kwargs):
        # about 4GB of VRAM required for 15000 pixel batch size (Total memory usage on GPU by the program).
        # about 4.6GB of VRAM required for 20000 pixel batch size.
        self.model = model
        self.device = device
        self.parameters = parameters
        self.scaler = scaler
        self.batch_size = batch_size
        self.kwargs = kwargs
        self._load()
        self._scaler()

    def _load(self):
        self.model = self.model(self.parameters["in_shape"], self.parameters['out_shape'])
        self.model.load_state_dict(torch.load(self.parameters['weights']))
        self.model.to(self.device)
        self.model.eval()

    def _scaler(self):
        if self.scaler == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler == "percentile":
            self.scaler = MinMaxScaler(percentiles=[2, 98])

    def predict(self, X, y=None):
        """
        Predicts the class labels for each pixel in X.
        Parameters
        ----------
        X : array-like, shape (width, height, wavelength)  # e.g. data from Cube
            The data to predict.
        Returns
        -------
        y : array-like, shape (width, height, 1)
            The predicted classes.
        """
        # reshape cube to model shape np.ndarray (pixels, wavelength)
        assert len(X.shape) == 3, "Please provide a 3D array"
        old_shape = X.shape
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        # create result array
        result = torch.full((X.shape[0], self.parameters["out_shape"]), 0, dtype=torch.float32).to(self.device)
        # # min max scale the data
        # X = self.scaler.process(X)
        X = vector_norm(X)  # model_V1 was trained on vector norm.
        # convert to torch tensor
        X = torch.from_numpy(X).float()
        # predict
        with torch.no_grad():
            for i in np.array_split(np.arange(X.shape[0]), X.shape[0] // self.batch_size):
                data = X[i, :].to(self.device)
                result[i, :] = self.model(data).data
        # reshape result to original shape
        result = result.cpu().numpy().astype(int)
        result = result.reshape(old_shape[0], old_shape[1], self.parameters["out_shape"])
        return result


def Main(vars):

    assert vars['parameters'] in MODEL_PARAMETERS, "Please provide a valid model parameter set"
    assert vars['device'] in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], "Please provide a valid device. Options: cpu, cuda, cuda:#"

    if vars['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    params = MODEL_PARAMETERS[opt['parameters']]
    cpc = CalibrationPlateClassifier(CalibrationModel, params, device, batch_size=vars['batch'])
    cube = Cube(vars["cube"])
    data = cube.data
    del cube  # delete the cube data from memory
    result = cpc.predict(data)
    np.save("data.npy", result)
    result = np.argmax(result, axis=-1).astype(np.uint8)
    plt.imshow(result)
    plt.axis('off')
    plt.savefig("results.png", bbox_inches='tight', pad_inches=0, dpi=500)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on. Default: cpu. Options: cpu, cuda.')
    parser.add_argument('--parameters', type=str, default='base_model', help='Model parameters. Default: base_model. Options: base_model.')
    parser.add_argument('--cube', type=str, help='Path to hyperspectral cube (.dat, or other ENVI format) file.', required=True)
    parser.add_argument('--batch', type=int, default=20000, help='Model inference batch size (pixels). Default: 20000.')
    # parser.add_argument('--scaler', type=str, default="vector_norm",
    #                     help='What scaler to use for hyperspectral data scaling. Default: minmax (used in model training).\
    #                           Options: minmax, percentile (2 and 98 percentiles if outliers are expected), vector_norm.')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    opt = vars(opt)
    Main(opt)
