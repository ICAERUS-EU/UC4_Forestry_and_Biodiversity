import numpy as np
import joblib
import argparse
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "weights", "forest_health_v1.pkl"))

EXPECTED_BANDS = 224

def fix_axes(img):
    # Find axis whose length matches 224
    axis_with_224 = [i for i, s in enumerate(img.shape) if s == EXPECTED_BANDS]

    if len(axis_with_224) == 1:
        band_axis = axis_with_224[0]
    else:
        # if there is no exact 224 match, assume largest axis is bands
        band_axis = np.argmax(img.shape)

    # Move bands to axis 0
    if band_axis != 0:
        img = np.moveaxis(img, band_axis, 0)

    # Now img shape is (B, H, W)
    # If B > 224, trim to expected count
    if img.shape[0] > EXPECTED_BANDS:
        img = img[:EXPECTED_BANDS]

    # If B < 224, model cannot run — raise error
    if img.shape[0] < EXPECTED_BANDS:
        raise ValueError(f"Not enough spectral bands. Need 224, got {img.shape[0]}.")

    return img

def save_npy(pred, out_path):
    np.save(out_path, pred)


def save_png(pred, out_path):
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(pred, cmap="viridis")
    plt.colorbar()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.5)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Forest health predictor")
    parser.add_argument("--npy", required=True)
    parser.add_argument("--save_npy")
    parser.add_argument("--save_png")
    return parser.parse_args()


def main():
    args = parse_args()

    base = os.path.splitext(args.npy)[0]
    if not args.save_npy and not args.save_png:
        args.save_npy = base + "_health_pred.npy"
        args.save_png = base + "_health_pred.png"

    reg = joblib.load(MODEL_PATH)

    img = np.load(args.npy)
    img = fix_axes(img)
    B, H, W = img.shape
    pred = reg.predict(img.reshape(B, -1).T).reshape(H, W)

    if args.save_npy:
        save_npy(pred, args.save_npy)
    if args.save_png:
        save_png(pred, args.save_png)


if __name__ == "__main__":
    main()
