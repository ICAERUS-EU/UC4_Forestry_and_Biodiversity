import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from indices import (
    CCI, ARI1, CRI1, SIPI, PRI, WBI, VRI1, PSSR, REP, YI
)

# Index processors
processors = {
    "CCI": CCI(),
    "ARI1": ARI1(),
    "CRI1": CRI1(),
    "SIPI": SIPI(),
    "PRI": PRI(),
    "WBI": WBI(),
    "VRI1": VRI1(),
    "PSSR": PSSR(),
    "REP": REP(),
    "YI": YI()
}

EXPECTED_BANDS = 224


def load_npy_cube(path):
    """Load and reorder a calibrated .npy hyperspectral cube."""

    if not path.lower().endswith(".npy"):
        raise ValueError("Indices mode only accepts calibrated .npy files.")

    arr = np.load(path)

    # Detect spectral band axis (largest dimension is usually bands)
    band_axis = np.argmax(arr.shape)

    # Move spectral bands to axis 0 → (B, H, W)
    arr = np.moveaxis(arr, band_axis, 0)

    # If more bands than needed, trim
    if arr.shape[0] > EXPECTED_BANDS:
        arr = arr[:EXPECTED_BANDS]

    # If fewer bands — cannot run
    if arr.shape[0] < EXPECTED_BANDS:
        raise ValueError(
            f"Input cube has only {arr.shape[0]} bands but requires {EXPECTED_BANDS}."
        )

    return arr


def save_npy(name, arr, out_dir):
    np.save(os.path.join(out_dir, f"{name}.npy"), arr)


def save_png(name, arr, out_dir):
    plt.figure(figsize=(30, 5))
    plt.axis("off")
    plt.imshow(arr, cmap="viridis")
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, f"{name}.png"),
                dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()


def run_indices(cube_path, out_dir, save_npy_flag, save_png_flag):
    X = load_npy_cube(cube_path)
    os.makedirs(out_dir, exist_ok=True)

    for name, proc in processors.items():
        result = proc(X)

        # Multi-value indices (PSSR returns 3)
        if isinstance(result, tuple):
            for i, subarr in enumerate(result):
                subname = f"{name}_{i+1}"
                if save_npy_flag:
                    save_npy(subname, subarr, out_dir)
                if save_png_flag:
                    save_png(subname, subarr, out_dir)
        else:
            if save_npy_flag:
                save_npy(name, result, out_dir)
            if save_png_flag:
                save_png(name, result, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Vegetation index calculator (npy only)")
    parser.add_argument("--cube", required=True, help=".npy calibrated hyperspectral cube")
    parser.add_argument("--out_dir", required=True, help="Where to save index outputs")
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    args = parser.parse_args()

    # Default: save both
    if not args.save_npy and not args.save_png:
        args.save_npy = True
        args.save_png = True

    run_indices(args.cube, args.out_dir, args.save_npy, args.save_png)


if __name__ == "__main__":
    main()
