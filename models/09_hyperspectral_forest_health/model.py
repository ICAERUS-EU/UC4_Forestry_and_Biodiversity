import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import rasterio

model_path = "forest_fuel.keras"

def load_data(data_path, a_coef, b_coef):
    cube = data_path
    with rasterio.open(cube) as src:
        img = src.read()
        crs = src.crs
    img = np.transpose(img, (1, 2, 0))
    a = np.load(a_coef)
    b = np.load(b_coef)
    img = img*a+b
    return img

def predict(model, data):
    results = np.zeros((data.shape[0], data.shape[1], 4))
    for i in range(data.shape[0]):
        chunk = data[i:i+1,:, :].squeeze()
        pred = model.predict(chunk)
        results[i, :, :] = pred
    np.save("predictions.npy", results)
    return results

def plot_predictions(predictions, output_path='predictions.png'):
    plt.axis("off")
    plt.imshow(np.argmax(predictions, axis=2).T)
    plt.savefig(output_path, dpi = 300)
    print(f"Saved plot to: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--output", default="predictions.png")
    parser.add_argument("a_coeficient_path")
    parser.add_argument("b_coeficient_path")
    args = parser.parse_args()
    print(args)

    model = tf.keras.models.load_model(model_path)
    data = load_data(args.data_path, args.a_coeficient_path, args.b_coeficient_path)
    pred = predict(model, data)
    plot_predictions(pred, args.output)

if __name__ == "__main__":
    main()
