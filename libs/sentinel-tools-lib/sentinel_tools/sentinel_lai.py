import math
import numpy as np
import xml.etree.ElementTree as ET
import numba
    

@numba.njit(parallel = True)
def tansig(data):
    """
    data[np.where(data < -10)] = -1.0
    data[np.where(data > 10)] = 1
    tmp = data[np.where((data >= -10) & (data <= 10))]
    data[np.where((data >= -10) & (data <= 10))] = 2 / (1 + np.exp(-2 * tmp)) - 1
    """
    return 2 / (1 + np.exp(-2 * data)) - 1


@numba.njit(parallel = True)
def normalize(data, min1, max1):
    return 2 * (data - min1) / (max1 - min1) - 1


@numba.njit(parallel = True)
def denormalize(data, min1, max1):
    return 0.5 * (data + 1) * (max1 - min1) + min1


def matrix_upscaler_500x(data):
    assert len(data.shape) == 2

    tmp = np.repeat(data, 500, axis=0)
    tmp500 = np.repeat(tmp, 500, axis=1)
    assert data.shape[0] * 500 == tmp500.shape[0]
    assert data.shape[1] * 500 == tmp500.shape[1]

    return tmp500

@numba.njit(parallel = True)
def neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):

    suma =  4.96238030555279  - 0.023406878966470 * b03_norm + \
    0.921655164636366 * b04_norm + 0.135576544080099 * b05_norm - \
    1.938331472397950 * b06_norm - 3.342495816122680 * b07_norm + \
    0.902277648009576 * b8a_norm + 0.205363538258614 * b11_norm - \
    0.040607844721716 * b12_norm - 0.083196409727092 * viewZen_norm + \
    0.260029270773809 * sunZen_norm + 0.284761567218845 * relAzim_norm 

    return tansig(suma)


def Neur1(n1, band, name):
    if name == "b03":
        return n1 - (0.023406878966470 * band)
    if name == "b04":
        return n1 + (0.921655164636366 * band)
    if name == "b05":
        return n1 + (0.135576544080099 * band)
    if name == "b06":
        return n1 - (1.938331472397950 * band)
    if name == "b07":
        return n1 - (3.342495816122680 * band)
    if name == "b08":
        return n1 + (0.902277648009576 * band)
    if name == "b11":
        return n1 + (0.205363538258614 * band)
    if name == "b12":
        return n1 - (0.040607844721716 * band)
    if name == "viewZen":
        return n1 - (0.083196409727092 * band)
    if name == "sunZen":
        return n1 + (0.260029270773809 * band)
    if name == "relAzim":
        return n1 + (0.284761567218845 * band)


@numba.njit(parallel = True)
def neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):
    
    suma = 1.416008443981500 - 0.132555480856684 * b03_norm - \
    0.139574837333540 * b04_norm - 1.014606016898920 * b05_norm - \
    1.330890038649270 * b06_norm + 0.031730624503341 * b07_norm - \
    1.433583541317050 * b8a_norm - 0.959637898574699 * b11_norm + \
    1.133115706551000 * b12_norm + 0.216603876541632 * viewZen_norm + \
    0.410652303762839 * sunZen_norm + 0.064760155543506 * relAzim_norm

    return tansig(suma)


def Neur2(n1, band, name):
    if name == "b03":
        return n1 - (0.132555480856684 * band)
    if name == "b04":
        return n1 - (0.139574837333540 * band)
    if name == "b05":
        return n1 - (1.014606016898920 * band)
    if name == "b06":
        return n1 - (1.330890038649270 * band)
    if name == "b07":
        return n1 + (0.031730624503341 * band)
    if name == "b08":
        return n1 - (1.433583541317050 * band)
    if name == "b11":
        return n1 - (0.959637898574699 * band)
    if name == "b12":
        return n1 + (1.133115706551000 * band)
    if name == "viewZen":
        return n1 + (0.216603876541632 * band)
    if name == "sunZen":
        return n1 + (0.410652303762839 * band)
    if name == "relAzim":
        return n1 + (0.064760155543506 * band)


@numba.njit(parallel = True)
def neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):

    suma = 1.075897047213310 + 0.086015977724868 * b03_norm + \
    0.616648776881434 * b04_norm + 0.678003876446556 * b05_norm + \
    0.141102398644968 * b06_norm - 0.096682206883546 * b07_norm - \
    1.128832638862200 * b8a_norm + 0.302189102741375 * b11_norm + \
    0.434494937299725 * b12_norm - 0.021903699490589 * viewZen_norm - \
    0.228492476802263 * sunZen_norm - 0.039460537589826 * relAzim_norm

    return tansig(suma)


def Neur3(n1, band, name):
    if name == "b03":
        return n1 + (0.086015977724868 * band)
    if name == "b04":
        return n1 + (0.616648776881434 * band)
    if name == "b05":
        return n1 + (0.678003876446556 * band)
    if name == "b06":
        return n1 + (0.141102398644968 * band)
    if name == "b07":
        return n1 - (0.096682206883546 * band)
    if name == "b08":
        return n1 - (1.128832638862200 * band)
    if name == "b11":
        return n1 + (0.302189102741375 * band)
    if name == "b12":
        return n1 + (0.434494937299725 * band)
    if name == "viewZen":
        return n1 - (0.021903699490589 * band)
    if name == "sunZen":
        return n1 - (0.228492476802263 * band)
    if name == "relAzim":
        return n1 - (0.039460537589826 * band)


@numba.njit(parallel = True)
def neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):

    suma = 1.533988264655420 - 0.109366593670404 * b03_norm - \
    0.071046262972729 * b04_norm + 0.064582411478320 * b05_norm + \
    2.906325236823160 * b06_norm - 0.673873108979163 * b07_norm - \
    3.838051868280840 * b8a_norm + 1.695979344531530 * b11_norm + \
    0.046950296081713 * b12_norm - 0.049709652688365 * viewZen_norm + \
    0.021829545430994 * sunZen_norm + 0.057483827104091 * relAzim_norm

    return tansig(suma)


def Neur4(n1, band, name):
    if name == "b03":
        return n1 - (0.109366593670404 * band)
    if name == "b04":
        return n1 - (0.071046262972729 * band)
    if name == "b05":
        return n1 + (0.064582411478320 * band)
    if name == "b06":
        return n1 + (2.906325236823160 * band)
    if name == "b07":
        return n1 - (0.673873108979163 * band)
    if name == "b08":
        return n1 - (3.838051868280840 * band)
    if name == "b11":
        return n1 + (1.695979344531530 * band)
    if name == "b12":
        return n1 + (0.046950296081713 * band)
    if name == "viewZen":
        return n1 - (0.049709652688365 * band)
    if name == "sunZen":
        return n1 + (0.021829545430994 * band)
    if name == "relAzim":
        return n1 + (0.057483827104091 * band)


@numba.njit(parallel = True)
def neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm):

    suma = 3.024115930757230 - 0.089939416159969 * b03_norm + \
    0.175395483106147 * b04_norm - 0.081847329172620 * b05_norm + \
    2.219895367487790 * b06_norm + 1.713873975136850 * b07_norm + \
    0.713069186099534 * b8a_norm + 0.138970813499201 * b11_norm - \
    0.060771761518025 * b12_norm + 0.124263341255473 * viewZen_norm + \
    0.210086140404351 * sunZen_norm - 0.183878138700341 * relAzim_norm

    return tansig(suma)


def Neur5(n1, band, name):
    if name == "b03":
        return n1 - (0.089939416159969 * band)
    if name == "b04":
        return n1 + (0.175395483106147 * band)
    if name == "b05":
        return n1 - (0.081847329172620 * band)
    if name == "b06":
        return n1 + (2.219895367487790 * band)
    if name == "b07":
        return n1 + (1.713873975136850 * band)
    if name == "b08":
        return n1 + (0.713069186099534 * band)
    if name == "b11":
        return n1 + (0.138970813499201 * band)
    if name == "b12":
        return n1 - (0.060771761518025 * band)
    if name == "viewZen":
        return n1 + (0.124263341255473 * band)
    if name == "sunZen":
        return n1 + (0.210086140404351 * band)
    if name == "relAzim":
        return n1 - (0.183878138700341 * band)


@numba.njit(parallel = True)
def layer2(neuron1, neuron2, neuron3, neuron4, neuron5):

    suma = 1.096963107077220 - 1.500135489728730 * neuron1 - 0.096283269121503 * neuron2 - 0.194935930577094 * neuron3 - 0.352305895755591 * neuron4 + 0.075107415847473 * neuron5
    
    return suma


def Lay2(l2, neur, name):
    if name == "n1":
        return l2 - (1.500135489728730 * neur)
    if name == "n2":
        return l2 - (0.096283269121503 * neur)
    if name == "n3":
        return l2 - (0.194935930577094 * neur)
    if name == "n4":
        return l2 - (0.352305895755591 * neur)
    if name == "n5":
        return l2 + (0.075107415847473 * neur)


def evaluate(b3, b4, b5, b6, b7, b8a, b11, b12, mean_zenith, mean_azimuth, zen, azim):

    degToRad = math.pi / 180
    main_shape = b4.shape
    assert len(main_shape) == 2
    viewZenithMean = mean_zenith
    viewAzimuthMean = mean_azimuth

    if b3.shape != main_shape:
        b3 = np.resize(b3, main_shape)
    if b4.shape != main_shape:
        b4 = np.resize(b4, main_shape)
    if b5.shape != main_shape:
        b5 = np.resize(b5, main_shape)
    if b6.shape != main_shape:
        b6 = np.resize(b6, main_shape)
    if b7.shape != main_shape:
        b7 = np.resize(b7, main_shape)
    if b8a.shape != main_shape:
        b8a = np.resize(b8a, main_shape)
    if b11.shape != main_shape:
        b11 = np.resize(b11, main_shape)
    if b12.shape != main_shape:
        b12 = np.resize(b12, main_shape)
    if azim.shape != main_shape:
        azim = np.resize(azim, main_shape)
    if zen.shape != main_shape:
        zen = np.resize(zen, main_shape)


    b03_norm = normalize(b3/10000, 0, 0.253061520471542)
    n1 = np.full(b03_norm.shape, 4.96238030555279)
    n2 = np.full(b03_norm.shape, 1.416008443981500)
    n3 = np.full(b03_norm.shape, 1.075897047213310)
    n4 = np.full(b03_norm.shape, 1.533988264655420)
    n5 = np.full(b03_norm.shape, 3.024115930757230)

    n1 = Neur1(n1, b03_norm, "b03")
    n2 = Neur2(n2, b03_norm, "b03")
    n3 = Neur3(n3, b03_norm, "b03")
    n4 = Neur4(n4, b03_norm, "b03")
    n5 = Neur5(n5, b03_norm, "b03")
    del b03_norm

    b04_norm = normalize(b4/10000, 0, 0.290393577911328)
    n1 = Neur1(n1, b04_norm, "b04")
    n2 = Neur2(n2, b04_norm, "b04")
    n3 = Neur3(n3, b04_norm, "b04")
    n4 = Neur4(n4, b04_norm, "b04")
    n5 = Neur5(n5, b04_norm, "b04")
    del b04_norm
    
    b05_norm = normalize(b5/10000, 0, 0.305398915248555)
    n1 = Neur1(n1, b05_norm, "b05")
    n2 = Neur2(n2, b05_norm, "b05")
    n3 = Neur3(n3, b05_norm, "b05")
    n4 = Neur4(n4, b05_norm, "b05")
    n5 = Neur5(n5, b05_norm, "b05")
    del b05_norm
    
    b06_norm = normalize(b6/10000, 0.006637972542253, 0.608900395797889)
    n1 = Neur1(n1, b06_norm, "b06")
    n2 = Neur2(n2, b06_norm, "b06")
    n3 = Neur3(n3, b06_norm, "b06")
    n4 = Neur4(n4, b06_norm, "b06")
    n5 = Neur5(n5, b06_norm, "b06")
    del b06_norm
    
    b07_norm = normalize(b7/10000, 0.013972727018939, 0.753827384322927)
    n1 = Neur1(n1, b07_norm, "b07")
    n2 = Neur2(n2, b07_norm, "b07")
    n3 = Neur3(n3, b07_norm, "b07")
    n4 = Neur4(n4, b07_norm, "b07")
    n5 = Neur5(n5, b07_norm, "b07")
    del b07_norm
    
    b8a_norm = normalize(b8a/10000, 0.026690138082061, 0.782011770669178)
    n1 = Neur1(n1, b8a_norm, "b08")
    n2 = Neur2(n2, b8a_norm, "b08")
    n3 = Neur3(n3, b8a_norm, "b08")
    n4 = Neur4(n4, b8a_norm, "b08")
    n5 = Neur5(n5, b8a_norm, "b08")
    del b8a_norm
    
    b11_norm = normalize(b11/10000, 0.016388074192258, 0.493761397883092)
    n1 = Neur1(n1, b11_norm, "b11")
    n2 = Neur2(n2, b11_norm, "b11")
    n3 = Neur3(n3, b11_norm, "b11")
    n4 = Neur4(n4, b11_norm, "b11")
    n5 = Neur5(n5, b11_norm, "b11")
    del b11_norm
    
    b12_norm = normalize(b12/10000, 0, 0.493025984460231)
    n1 = Neur1(n1, b12_norm, "b12")
    n2 = Neur2(n2, b12_norm, "b12")
    n3 = Neur3(n3, b12_norm, "b12")
    n4 = Neur4(n4, b12_norm, "b12")
    n5 = Neur5(n5, b12_norm, "b12")
    del b12_norm
    
    viewZen_norm = normalize(np.cos(zen * degToRad), 0.918595400582046, 1)
    n1 = Neur1(n1, viewZen_norm, "viewZen")
    n2 = Neur2(n2, viewZen_norm, "viewZen")
    n3 = Neur3(n3, viewZen_norm, "viewZen")
    n4 = Neur4(n4, viewZen_norm, "viewZen")
    n5 = Neur5(n5, viewZen_norm, "viewZen")
    del viewZen_norm
    
    sunZen_norm  = normalize(np.cos(azim * degToRad), 0.342022871159208, 0.936206429175402)
    n1 = Neur1(n1, sunZen_norm, "sunZen")
    n2 = Neur2(n2, sunZen_norm, "sunZen")
    n3 = Neur3(n3, sunZen_norm, "sunZen")
    n4 = Neur4(n4, sunZen_norm, "sunZen")
    n5 = Neur5(n5, sunZen_norm, "sunZen")
    del sunZen_norm
    
    relAzim_norm = np.cos((zen - azim) * degToRad)
    n1 = Neur1(n1, relAzim_norm, "relAzim")
    n2 = Neur2(n2, relAzim_norm, "relAzim")
    n3 = Neur3(n3, relAzim_norm, "relAzim")
    n4 = Neur4(n4, relAzim_norm, "relAzim")
    n5 = Neur5(n5, relAzim_norm, "relAzim")
    del relAzim_norm

    """
    n1 = neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n2 = neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n3 = neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n4 = neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    n5 = neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm)
    """
    #l2 = layer2(n1, n2, n3, n4, n5)
    l2 = np.full(n1.shape, 1.096963107077220)
    l2 = Lay2(l2, tansig(n1), "n1")
    del n1

    l2 = Lay2(l2, tansig(n2), "n2")
    del n2
    
    l2 = Lay2(l2, tansig(n3), "n3")
    del n3
    
    l2 = Lay2(l2, tansig(n4), "n4")
    del n4

    l2 = Lay2(l2, tansig(n5), "n5")
    del n5

    vals = denormalize(l2, 0.000319182538301, 14.4675094548151) / 3
    return vals


def main(b3, b4, b5, b6, b7, b8a, b11, b12, zen, azim):
    a = np.average(zen)
    b = np.average(azim)

    return evaluate(b3, b4, b5, b6, b7, b8a, b11, b12, a, b, zen, azim)


