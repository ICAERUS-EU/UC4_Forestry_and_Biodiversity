Fuel

--mode fuel
--input <input_file.npy>
--output_npy <prediction_output.npy>   (default: <input>_fuel_pred.npy)
--output_png <prediction_output.png>   (default: <input>_fuel_pred.png)

python run.py \
    --mode fuel \
    --input calibrated_s.npy \
    --output_npy fuel_pred.npy \
    --output_png fuel_pred.png

Health

--mode health
--input <input_file.npy>
--output_npy <prediction_output.npy>   (default: <input>_health_pred.npy)
--output_png <prediction_output.png>   (default: <input>_health_pred.png)

python run.py \
    --mode health \
    --input calibrated_s.npy \
    --output_npy health_pred.npy \
    --output_png health_pred.png

Indices

--mode indices
--input <input_file.npy>
--out_dir <folder_to_save_outputs>
--save_npy     (default = ON if no flags given)
--save_png     (default = ON if no flags given)

python run.py \
    --mode indices \
    --input input.npy \
    --out_dir indices_output \
    --save_npy \
    --save_png

