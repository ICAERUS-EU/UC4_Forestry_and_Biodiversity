import argparse
import subprocess
import sys
import os

# Add module folders to Python path
sys.path.append("09_hyperspectral_forest_health")
sys.path.append("10_hyperspectral_forest_fuel")


def run_fuel_model(input_path, npy_out, png_out):
    cmd = ["python3", "10_hyperspectral_forest_fuel/model_fuel.py", input_path]
    if npy_out:
        cmd.extend(["--save_npy", npy_out])
    if png_out:
        cmd.extend(["--save_png", png_out])
    subprocess.run(cmd, check=True)


def run_health_model(input_path, npy_out, png_out):
    cmd = ["python3", "09_hyperspectral_forest_health/model_health.py", "--npy", input_path]
    if npy_out:
        cmd.extend(["--save_npy", npy_out])
    if png_out:
        cmd.extend(["--save_png", png_out])
    subprocess.run(cmd, check=True)


def run_indices_model(input_path, out_dir, save_npy, save_png):
    cmd = ["python3", "09_hyperspectral_forest_health/indices_runner.py",
           "--cube", input_path, "--out_dir", out_dir]
    if save_npy:
        cmd.append("--save_npy")
    if save_png:
        cmd.append("--save_png")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Unified runner")
    parser.add_argument("--mode", required=True, choices=["fuel", "health", "indices"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_npy")
    parser.add_argument("--output_png")
    parser.add_argument("--out_dir")
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    args = parser.parse_args()

    if args.mode in ["fuel", "health"]:
        if not args.output_npy and not args.output_png:
            base = args.input.rsplit(".", 1)[0]
            if args.mode == "fuel":
                args.output_npy = base + "_fuel_pred.npy"
                args.output_png = base + "_fuel_pred.png"
            else:
                args.output_npy = base + "_health_pred.npy"
                args.output_png = base + "_health_pred.png"

    if args.mode == "fuel":
        run_fuel_model(args.input, args.output_npy, args.output_png)

    elif args.mode == "health":
        run_health_model(args.input, args.output_npy, args.output_png)

    elif args.mode == "indices":
        if not args.out_dir:
            raise ValueError("--out_dir is required for indices mode")

        if not args.save_npy and not args.save_png:
            args.save_npy = True
            args.save_png = True

        run_indices_model(args.input, args.out_dir, args.save_npy, args.save_png)


if __name__ == "__main__":
    main()
