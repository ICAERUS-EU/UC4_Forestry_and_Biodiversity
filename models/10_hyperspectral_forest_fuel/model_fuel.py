import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "weights", "forest_fuel.keras")
model_path = os.path.abspath(model_path)



def load_data(npy_path):
    arr = np.load(npy_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got {arr.shape}")
    if arr.shape[0] < 10:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def predict(model, data):
    H, W, B = data.shape
    results = np.zeros((H, W, 4))
    for i in range(H):
        row = data[i:i+1, :, :].squeeze()
        pred = model.predict(row, verbose=0)
        results[i, :, :] = pred
    return results


def save_npy(pred, out_path):
    np.save(out_path, pred)


def save_png(pred, out_path):
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(np.argmax(pred, axis=2).T)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Forest fuel model")
    parser.add_argument("npy")
    parser.add_argument("--save_npy")
    parser.add_argument("--save_png")
    args = parser.parse_args()

    base = os.path.splitext(args.npy)[0]
    if not args.save_npy and not args.save_png:
        args.save_npy = base + "_fuel_pred.npy"
        args.save_png = base + "_fuel_pred.png"

    model = tf.keras.models.load_model(model_path)
    data = load_data(args.npy)
    pred = predict(model, data)

    if args.save_npy:
        save_npy(pred, args.save_npy)
    if args.save_png:
        save_png(pred, args.save_png)


if __name__ == "__main__":
    main()
