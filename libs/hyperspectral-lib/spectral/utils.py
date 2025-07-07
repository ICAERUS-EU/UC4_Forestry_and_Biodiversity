import numpy as np


SPECIM_WAVELENGTHS_2X1 = np.array([397.66, 400.28, 402.9, 405.52, 408.13, 410.75, 413.37, 416, 418.62, 421.24, 423.86, 426.49, 429.12, 431.74, 434.37, 437, 439.63, 442.26, 444.89, 447.52, 450.16,
                                   452.79, 455.43, 458.06, 460.7, 463.34, 465.98, 468.62, 471.26, 473.9, 476.54, 479.18, 481.83, 484.47, 487.12, 489.77, 492.42, 495.07, 497.72, 500.37, 503.02, 505.67,
                                   508.32, 510.98, 513.63, 516.29, 518.95, 521.61, 524.27, 526.93, 529.59, 532.25, 534.91, 537.57, 540.24, 542.91, 545.57, 548.24, 550.91, 553.58, 556.25, 558.92,
                                   561.59, 564.26, 566.94, 569.61, 572.29, 574.96, 577.64, 580.32, 583, 585.68, 588.36, 591.04, 593.73, 596.41, 599.1, 601.78, 604.47, 607.16, 609.85, 612.53, 615.23,
                                   617.92, 620.61, 623.3, 626, 628.69, 631.39, 634.08, 636.78, 639.48, 642.18, 644.88, 647.58, 650.29, 652.99, 655.69, 658.4, 661.1, 663.81, 666.52, 669.23, 671.94,
                                   674.65, 677.36, 680.07, 682.79, 685.5, 688.22, 690.93, 693.65, 696.37, 699.09, 701.81, 704.53, 707.25, 709.97, 712.7, 715.42, 718.15, 720.87, 723.6, 726.33, 729.06,
                                   731.79, 734.52, 737.25, 739.98, 742.72, 745.45, 748.19, 750.93, 753.66, 756.4, 759.14, 761.88, 764.62, 767.36, 770.11, 772.85, 775.6, 778.34, 781.09, 783.84, 786.58,
                                   789.33, 792.08, 794.84, 797.59, 800.34, 803.1, 805.85, 808.61, 811.36, 814.12, 816.88, 819.64, 822.4, 825.16, 827.92, 830.69, 833.45, 836.22, 838.98, 841.75, 844.52,
                                   847.29, 850.06, 852.83, 855.6, 858.37, 861.14, 863.92, 866.69, 869.47, 872.25, 875.03, 877.8, 880.58, 883.37, 886.15, 888.93, 891.71, 894.5, 897.28, 900.07, 902.86,
                                   905.64, 908.43, 911.22, 914.02, 916.81, 919.6, 922.39, 925.19, 927.98, 930.78, 933.58, 936.38, 939.18, 941.98, 944.78, 947.58, 950.38, 953.19, 955.99, 958.8, 961.6,
                                   964.41, 967.22, 970.03, 972.84, 975.65, 978.46, 981.27, 984.09, 986.9, 989.72, 992.54, 995.35, 998.17, 1000.99, 1003.81])


def wavelength_to_band(wl, wls=None):
    """
    Return the index of the closest band given the wavelength

    Parameters:
        wl: float
            Wavelength in nanometers.

        wls: np.array of list
            Camera wavelenghts for each index
    """
    if wls is None:
        wls = SPECIM_WAVELENGTHS_2X1
    if type(wls) is list:
        wls = np.array(wls)
    band = np.argmin(np.abs(wls - wl))
    return band


def wavelength_match(wl1, wl2, reduce=True):  # todo: Normal testing
    """
    Parameters
        wl1 : np.array or list
            1D array or list of size N of wavelength that needed to be matched

        wl2 : np.array or list
            1D array or list of size M

        reduce : boolean
            Set true if reduction to the smallest number of wavelenghts is required, otherwise extend (by duplication) to bigger number

    Returns
        indx1 : np.array
            1D array of final index values to use for wl1

        indx2 : np.array
            2D array of final index values to use for wl2

    """
    assert type(wl1) is list or type(wl1) is np.ndarray, "wl1 has te be of type list or numpy array"
    assert type(wl2) is list or type(wl2) is np.ndarray, "wl2 has te be of type list or numpy array"

    if type(wl1) is list:
        wl1 = np.array(wl1)
    if type(wl2) is list:
        wl2 = np.array(wl2)

    wl1 = np.squeeze(wl1)
    wl2 = np.squeeze(wl2)

    assert len(wl1.shape) == 1 and len(wl2.shape) == 1, "wavelenght arrays need to be 1D"

    if wl1.shape[0] >= wl2.shape[0]:  # if wl1 is bigger
        bigger = True
    else:
        bigger = False

    if reduce:  # match to the smaller array
        if bigger:  # if first one is bigger
            rez1 = []
            for i in range(wl2.shape[0]):  # wl1 bigger match wl1 indx to all wl2 indx
                rez1.append(np.argmin(np.abs(wl1 - wl2[i])))
            rez2 = np.arange(wl2.shape[0])
            return rez1, rez2
        else:  # if second one is bigger
            rez2 = []
            for i in range(wl1.shape[0]):
                rez2.append(np.argmin(np.abs(wl2 - wl1[i])))
            rez1 = np.arange(wl1.shape[0])
            return rez1, rez2
    else:
        # WIP if it is required at all
        pass


def spectral_calculator(data, axis=-1, method="average"):
    """
    Simple numpy spectral data calculator

    Parameters
        data : np.array
            2D input data

        axis : int
            integer to indicate on which axis to perform calculation

        method : str
            string to determine the method to use. Available methods: average, median
    """
    if method == "average":
        return np.average(data, axis=axis)
    elif method == "median":
        return np.median(data, axis=axis)
    else:
        print("method not found or incorrect, returning data")
        return data


def calibrate(data, calibration_spectra, reflection_percentages, dark_cal=None, wavelengths=None, wl_axis=-1, method="average", calibration_curve_degree=1):
    """
    Calculate cube reflectance from given DRS (Diffuse Reflectance Standard) spectra.
    As a base this was used: https://d-nb.info/1216632197/34
    Main formula R = (raw - dark) / (white - dark)

    Parameters
        data : np.array shape -> (X, Y, WL)
            hyperspectral input data
        calibration_spectra : np.array shape -> (N, WL), (N, M, WL)
            input of calibration spectra in singular array
        reflectance_percentages : np.array shape -> (N), (N, WL)
            the reflectance percentager for each of calibration spectra, maybe different for each wavelength
        dark_cal : np.array shape -> (N, WL), (N, M, WL)
            the reflectance of value 0 (closed camera captured data)
        wavelenghts : list
            List of wavelengths, used only in assertions and is optional
        wl_axis : int
            which calibration spectra axis is the wavelenght axis
        method : str
            which method to use for calibraion spectra aggregation if required.
        calibration_curve_degree : int
            degree of polynomial to fit to the calibration curve. Use 1 for black and white reflectance only, with more calibration curves higher degree may be more accurate.
    """
    shp1 = calibration_spectra.shape
    shp2 = reflection_percentages.shape

    if calibration_curve_degree > 2:
        print("Higher degree polynomials are WIP")
        return False

    assert len(shp1) == 2 or len(shp1) == 3, "calibration_spectra shape is out range"
    assert len(shp2) == 1 or len(shp2) == 2, "reflection_percentages shape is out range"
    assert type(wl_axis) is int, "wl_axis is not an integer"

    if wavelengths is not None:
        if len(shp2) == 2:
            assert len(shp2[0]) == len(wavelengths) or len(shp2[1]) == len(wavelengths), "mismatch between wavelenght counts in cube and relection_precentages data"
        assert shp1[wl_axis] == len(wavelengths), "mismatch between wavelength of calibration spectra and data cube"
    # dark check
    if dark_cal is not None:
        if len(shp1) == 2:
            calibration_spectra = np.append(calibration_spectra, dark_cal, axis=0)
            reflection_percentages = np.append(reflection_percentages, np.array([0]), axis=0)
            # update shapes
            shp1 = calibration_spectra.shape
            shp2 = reflection_percentages.shape
        else:
            # WIP
            return False

    # get reflectance calibration functions for each wavelength
    calib = []
    for wl in range(shp1[wl_axis]):
        cal_spectra = np.take(calibration_spectra, wl, axis=wl_axis)
        if len(cal_spectra.shape) > 1:
            cal_spectra = spectral_calculator(cal_spectra, wl_axis - 1 if wl_axis > 0 else wl_axis, method)
        if len(cal_spectra.shape) == 2:
            z = np.polyfit(cal_spectra[:, wl], reflection_percentages[:, wl], calibration_curve_degree)
        else:
            z = np.polyfit(cal_spectra, reflection_percentages, calibration_curve_degree)
        calib.append(z.copy())
    calib = np.array(calib)

    # calibrate data
    data = data.astype("float")
    if calib.shape[-1] == 3:  # quadratic
        for wl in range(shp1[wl_axis]):
            tmp = data[..., wl]
            data[..., wl] = tmp * tmp * calib[wl, 0] + tmp * calib[wl, 1] + calib[wl, 2]
    elif calib.shape[-1] == 2:  # linear
        for wl in range(shp1[wl_axis]):
            tmp = data[..., wl]
            data[..., wl] = tmp * calib[wl, 0] + calib[wl, 1]
    return data
