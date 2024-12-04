import numpy as np


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
    assert type(wl1) == list or type(wl1) == np.ndarray, "wl1 has te be of type list or numpy array"
    assert type(wl2) == list or type(wl2) == np.ndarray, "wl2 has te be of type list or numpy array"

    if type(wl1) == list:
        wl1 = np.array(wl1)
    if type(wl2) == list:
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
    assert type(wl_axis) == int, "wl_axis is not an integer"
    
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

