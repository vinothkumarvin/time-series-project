
import os
import json
import math
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Reproducibility seeds
SEED = 42
np.random.seed(SEED)

# Machine learning & stats
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import StandardScaler
import optuna
from statsmodels.tsa.statespace.sarimax import SARIMAX
def generate_multivariate_series(
    n_steps: int = 3000,
    n_series: int = 6,
    freq_components: List[Tuple[int, float]] = None,
    hetero_coeff: float = 0.5,
    rng_seed: int = SEED,
) -> pd.DataFrame:

    rng = np.random.default_rng(rng_seed)
    t = np.arange(n_steps)
    df = pd.DataFrame({"t": t})

    # Nonlinear trend shared across features (but scaled per feature)
    trend = 0.0006 * (t ** 1.5)  # gentle nonlinear upward trend

    if freq_components is None:
        # default: approximate daily, weekly, yearly cycles (if time unit is hours/days adjust accordingly)
        freq_components = [(24, 1.0), (168, 0.6), (365, 0.3)]

    for i in range(n_series):
        series = trend * (1 + 0.08 * i)

        # Add several sinusoidal components with random phases and small random amplitude modulation
        for period, amp in freq_components:
            phase = rng.uniform(0, 2 * np.pi)
            mod = 1 + 0.2 * rng.normal(size=n_steps)
            series = series + amp * mod * np.sin(2 * np.pi * (t / period) + phase)

        # Feature-specific mild AR-like component (constructed deterministically)
        phi = 0.6 - 0.12 * (i / max(1, n_series - 1))
        ar = np.zeros(n_steps)
        noise_for_ar = rng.normal(scale=0.5, size=n_steps)
        for k in range(1, 6):
            ar[k:] += (phi ** k) * noise_for_ar[:-k]
        series = series + 0.4 * ar

        # Heteroscedastic noise: noise scale increases with absolute signal
        base_std = 0.3 + 0.05 * i
        scale = hetero_coeff * (1 + 0.3 * np.abs(series) / (np.std(series) + 1e-9))
        noise = rng.normal(scale=base_std * scale)
        series = series + noise

        df[f"feat_{i}"] = series

    return df
class Seq2SeqPreprocessor:


    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.fitted = False

    def fit(self, data: np.ndarray, train_end_idx: int):

        assert train_end_idx > 0 and train_end_idx <= data.shape[0]
        self.scaler.fit(data[:train_end_idx])
        self.fitted = True

    def transform(self, data: np.ndarray, input_steps: int, output_steps: int):
        """
        Return:
            X_enc: (N, input_steps, F)
            X_dec: (N, output_steps, F) -> teacher forcing inputs (shifted targets with initial zeros)
            Y:     (N, output_steps, F)
        """
        assert self.fitted, "Call fit() before transform()"
        Xs = self.scaler.transform(data)
        T, F = Xs.shape
        seq_count = T - input_steps - output_steps + 1
        X_enc = np.zeros((seq_count, input_steps, F), dtype=np.float32)
        X_dec = np.zeros((seq_count, output_steps, F), dtype=np.float32)
        Y = np.zeros((seq_count, output_steps, F), dtype=np.float32)

        for i in range(seq_count):
            X_enc[i] = Xs[i:i + input_steps]
            Y[i] = Xs[i + input_steps:i + input_steps + output_steps]
            # decoder teacher forcing input: shift Y right by 1, start with zeros
            X_dec[i, 1:] = Y[i, :-1]
            X_dec[i, 0] = 0.0

        return X_enc, X_dec, Y

    def inverse_transform(self, scaled_array: np.ndarray):
        """
        scaled_array: (..., F)
        returns same shape, inverse scaled
        """
        shape = scaled_array.shape
        flat = scaled_array.reshape(-1, shape[-1])
        inv = self.scaler.inverse_transform(flat)
        return inv.reshape(shape)

def build_attention_lstm_model(
    input_steps: int,
    output_steps: int,
    n_features: int,
    lstm_units: int = 128,
    attention_units: int = 64,
    dropout: float = 0.2,
    enc_layers: int = 1,
    dec_layers: int = 1,
) -> tf.keras.Model:


    # Encoder
    encoder_inputs = layers.Input(shape=(input_steps, n_features), name="encoder_inputs")
    x = encoder_inputs
    for _ in range(enc_layers):
        x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(x)
    encoder_outputs = x  # shape (batch, enc_steps, lstm_units)

    # Decoder
    decoder_inputs = layers.Input(shape=(output_steps, n_features), name="decoder_inputs")
    y = decoder_inputs
    # Project decoder inputs to LSTM units dimension before feeding (optional but helps)
    y = layers.TimeDistributed(layers.Dense(lstm_units))(y)
    for _ in range(dec_layers):
        y = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)(y)
    decoder_outputs = y  # shape (batch, dec_steps, lstm_units)

    # Attention: compute context for each decoder time step
    # Keras Attention expects (query, value). We'll use query=decoder_outputs, value=encoder_outputs
    attention = layers.Attention(name="attention_layer")([decoder_outputs, encoder_outputs])
    # attention shape: (batch, dec_steps, enc_units) where enc_units == lstm_units

    # Combine decoder outputs with attention context
    combined = layers.Concatenate(axis=-1)([decoder_outputs, attention])  # (batch, dec_steps, lstm_units*2)

    # Optionally project through a dense + dropout
    proj = layers.TimeDistributed(layers.Dense(attention_units, activation="tanh"))(combined)
    proj = layers.Dropout(dropout)(proj)

    outputs = layers.TimeDistributed(layers.Dense(n_features), name="output_projection")(proj)

    model = models.Model([encoder_inputs, decoder_inputs], outputs)
    return model

def sarimax_forecast_univariate(train_series: np.ndarray, steps: int, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
  
  
    train_series = np.asarray(train_series).astype(float).reshape(-1)
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, method='lbfgs', maxiter=100)
    preds = fit.forecast(steps=steps)
    return np.asarray(preds)

def sarimax_forecast_multivariate(train_data: np.ndarray, steps: int, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
 
    T, F = train_data.shape
    preds = np.zeros((steps, F), dtype=float)
    for f in range(F):
        preds[:, f] = sarimax_forecast_univariate(train_data[:, f], steps, order=order, seasonal_order=seasonal_order)
    return preds


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 1) -> float:
  
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)
    # Reshape to 2D: (M, F)
    def flatten_2d(arr):
        if arr.ndim == 3:
            N, h, F = arr.shape
            return arr.reshape(N * h, F)
        elif arr.ndim == 2:
            return arr
        elif arr.ndim == 1:
            return arr.reshape(-1, 1)
        else:
            raise ValueError("Unexpected array shape for MASE.")
    Yt = flatten_2d(y_true)
    Yp = flatten_2d(y_pred)
    T, F = y_train.shape
    # denominator: mean absolute difference of naive seasonal forecast on training set
    if seasonality >= T:
        # fall back to first-difference naive
        denom = np.mean(np.abs(y_train[1:] - y_train[:-1]), axis=0)
    else:
        denom = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]), axis=0)
    denom = np.where(denom == 0, 1e-9, denom)  # avoid div by zero
    num = np.mean(np.abs(Yt - Yp), axis=0)
    per_feat = num / denom
    return float(np.mean(per_feat))

def rolling_origin_splits(T: int, train_window: int, fh: int, step: int = 1):
  
    start = train_window
    while start + fh <= T:
        train_idx = slice(0, start)
        val_idx = slice(start, start + fh)
        yield train_idx, val_idx
        start += step


def optuna_objective_factory(
    data: np.ndarray,
    input_steps: int,
    output_steps: int,
    train_initial_window: int,
    max_eval_windows: int = 3,
    timeout_minutes: int = None,
):
   

    def objective(trial):
        # Hyperparameters to search
        lstm_units = trial.suggest_int("lstm_units", 32, 192, step=16)
        attention_units = trial.suggest_int("attention_units", 16, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        lr = trial.suggest_loguniform("learning_rate", 1e-4, 5e-3)
        enc_layers = trial.suggest_int("enc_layers", 1, 2)
        dec_layers = trial.suggest_int("dec_layers", 1, 2)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "nadam", "rmsprop"])

        # Build model function factory (so we can remake per trial)
        def build_model():
            m = build_attention_lstm_model(
                input_steps=input_steps,
                output_steps=output_steps,
                n_features=data.shape[1],
                lstm_units=lstm_units,
                attention_units=attention_units,
                dropout=dropout,
                enc_layers=enc_layers,
                dec_layers=dec_layers,
            )
            opt_cls = {"adam": optimizers.Adam, "nadam": optimizers.Nadam, "rmsprop": optimizers.RMSprop}[optimizer_name]
            m.compile(optimizer=opt_cls(learning_rate=lr), loss="mse")
            return m

        # Do a limited rolling-origin evaluation using last windows for speed
        maes = []
        splits = list(rolling_origin_splits(T=data.shape[0], train_window=train_initial_window, fh=output_steps, step=output_steps))
        # Keep last `max_eval_windows` splits (or all if fewer)
        eval_splits = splits[-max_eval_windows:] if len(splits) > 0 else []
        if not eval_splits:
            # fallback: do a single split
            eval_splits = [(slice(0, train_initial_window), slice(train_initial_window, train_initial_window + output_steps))]

        for train_idx, val_idx in eval_splits:
            # fit scaler on train portion
            pre = Seq2SeqPreprocessor()
            pre.fit(data, train_end_idx=train_idx.stop)
            X_enc, X_dec, Y = pre.transform(data, input_steps=input_steps, output_steps=output_steps)
            # locate which sequences correspond to val_idx
            # sequence i corresponds to target at timesteps [i+input_steps, i+input_steps+output_steps)
            seq_start = 0
            seq_end = X_enc.shape[0]
            # identify sequence indices where target starts at train_idx.stop
            # find i such that i + input_steps == train_idx.stop
            val_seq_i = train_idx.stop - input_steps
            if val_seq_i < 0 or val_seq_i + 1 > seq_end:
                # if mismatch due to small sizes, skip this split
                continue

            # Training data: all sequences strictly before val_seq_i
            X_train_enc = X_enc[:val_seq_i]
            X_train_dec = X_dec[:val_seq_i]
            Y_train = Y[:val_seq_i]
            # Validation sequence: the single sequence that starts at train_idx.stop
            X_val_enc = X_enc[val_seq_i:val_seq_i + 1]
            X_val_dec = X_dec[val_seq_i:val_seq_i + 1]
            Y_val = Y[val_seq_i:val_seq_i + 1]

            if X_train_enc.shape[0] < 4:
                # not enough training sequences for this split; skip
                continue

            model = build_model()
            es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
            model.fit([X_train_enc, X_train_dec], Y_train, validation_data=([X_val_enc, X_val_dec], Y_val),
                      epochs=60, batch_size=batch_size, callbacks=[es], verbose=0)

            preds = model.predict([X_val_enc, X_val_dec])
            # Inverse scale
            preds_inv = pre.inverse_transform(preds[0])  # shape (output_steps, F)
            y_val_inv = pre.inverse_transform(Y_val[0])
            y_train_inv = pre.inverse_transform(data[train_idx])  # training raw (T_train, F)
            # compute MASE
            m = mase(y_true=y_val_inv[np.newaxis, ...], y_pred=preds_inv[np.newaxis, ...], y_train=y_train_inv, seasonality=1)
            maes.append(m)

        # If no maes collected, return large number
        if not maes:
            return 1e6
        return float(np.mean(maes))

    return objective


def train_and_evaluate_full(
    data: np.ndarray,
    input_steps: int,
    output_steps: int,
    best_params: dict,
    train_end_idx: int,
    rolling_eval_splits: List[Tuple[slice, slice]] = None,
    save_dir: str = "./results",
)
    os.makedirs(save_dir, exist_ok=True)
    # Prepare preprocessor fitted on train_end_idx
    pre = Seq2SeqPreprocessor()
    pre.fit(data, train_end_idx=train_end_idx)

    X_enc_all, X_dec_all, Y_all = pre.transform(data, input_steps=input_steps, output_steps=output_steps)
    # Determine train/val split indices at sequence level: use sequences with target starting < train_end_idx for training
    seq_count = X_enc_all.shape[0]
    # sequence i's target starts at i + input_steps
    train_seq_end = train_end_idx - input_steps  # exclusive
    train_seq_end = max(1, min(train_seq_end, seq_count - 1))
    X_train_enc = X_enc_all[:train_seq_end]
    X_train_dec = X_dec_all[:train_seq_end]
    Y_train = Y_all[:train_seq_end]

    # Build model with best params
    model = build_attention_lstm_model(
        input_steps=input_steps,
        output_steps=output_steps,
        n_features=data.shape[1],
        lstm_units=int(best_params.get("lstm_units", 128)),
        attention_units=int(best_params.get("attention_units", 64)),
        dropout=float(best_params.get("dropout", 0.2)),
        enc_layers=int(best_params.get("enc_layers", 1)),
        dec_layers=int(best_params.get("dec_layers", 1)),
    )
    opt_name = best_params.get("optimizer", "adam")
    lr = float(best_params.get("learning_rate", 1e-3))
    opt_cls = {"adam": optimizers.Adam, "nadam": optimizers.Nadam, "rmsprop": optimizers.RMSprop}[opt_name]
    model.compile(optimizer=opt_cls(learning_rate=lr), loss="mse")

    es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    # Use last 10% of training sequences for validation during final training
    val_split = max(1, int(0.1 * X_train_enc.shape[0]))
    model.fit([X_train_enc[:-val_split], X_train_dec[:-val_split]], Y_train[:-val_split],
              validation_data=([X_train_enc[-val_split:], X_train_dec[-val_split:]], Y_train[-val_split:]),
              epochs=150, batch_size=int(best_params.get("batch_size", 64)), callbacks=[es], verbose=1)

    # Save model architecture + weights
    model_path = os.path.join(save_dir, "attention_lstm_model.h5")
    model.save(model_path, include_optimizer=False)

    # Save best_params
    params_path = os.path.join(save_dir, "best_hyperparameters.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Evaluate on rolling-origin splits if provided, otherwise evaluate the single holdout window directly after train_end_idx
    if rolling_eval_splits is None:
        # single evaluation: predict one window right after train_end_idx
        val_start = train_end_idx
        val_slice = slice(val_start, val_start + output_steps)
        train_slice = slice(0, train_end_idx)
        rolling_eval_splits = [(train_slice, val_slice)]

    results = {"dl_mase": [], "sarimax_mase": [], "details": []}

    for train_slice, val_slice in rolling_eval_splits:
        # Prepare sequences for the origin at train_slice.stop
        pre_local = Seq2SeqPreprocessor()
        pre_local.fit(data, train_end_idx=train_slice.stop)
        X_enc_local, X_dec_local, Y_local = pre_local.transform(data, input_steps=input_steps, output_steps=output_steps)
        seq_i = train_slice.stop - input_steps
        if seq_i < 0 or seq_i >= X_enc_local.shape[0]:
            continue
        X_val_enc = X_enc_local[seq_i:seq_i + 1]
        X_val_dec = X_dec_local[seq_i:seq_i + 1]
        Y_val = Y_local[seq_i:seq_i + 1]

        # DL predictions
        dl_preds_scaled = model.predict([X_val_enc, X_val_dec])  # (1, h, F)
        dl_preds = pre_local.inverse_transform(dl_preds_scaled[0])  # (h, F)
        y_val = pre_local.inverse_transform(Y_val[0])  # (h, F)
        y_train = pre_local.inverse_transform(data[train_slice])  # (T_train, F)

        dl_m = mase(y_true=y_val[np.newaxis, ...], y_pred=dl_preds[np.newaxis, ...], y_train=y_train, seasonality=1)

        # SARIMAX baseline: fit on raw training slice and forecast h steps
        sarimax_preds = sarimax_forecast_multivariate(train_data=y_train, steps=output_steps)
        sar_m = mase(y_true=y_val[np.newaxis, ...], y_pred=sarimax_preds[np.newaxis, ...], y_train=y_train, seasonality=1)

        results["dl_mase"].append(dl_m)
        results["sarimax_mase"].append(sar_m)
        results["details"].append({
            "train_slice": (train_slice.start, train_slice.stop),
            "val_slice": (val_slice.start, val_slice.stop),
            "dl_mase": dl_m,
            "sarimax_mase": sar_m
        })

    # aggregate results
    results["dl_mase_mean"] = float(np.mean(results["dl_mase"])) if results["dl_mase"] else None
    results["sarimax_mase_mean"] = float(np.mean(results["sarimax_mase"])) if results["sarimax_mase"] else None

    # Save results summary
    summary_path = os.path.join(save_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_full_experiment(
    n_steps=3000,
    n_series=6,
    input_steps=48,
    output_steps=24,
    initial_train_window=2000,
    opt_trials=24,
    opt_timeout_minutes=None,
    results_dir="./results",
):
   

    # 1) Data
    df = generate_multivariate_series(n_steps=n_steps, n_series=n_series)
    features = [c for c in df.columns if c.startswith("feat_")]
    data = df[features].values  # shape (T, F)

    # 2) Optuna search
    study_name = "attention_lstm_opt"
    storage = None
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name)
    objective = optuna_objective_factory(
        data=data,
        input_steps=input_steps,
        output_steps=output_steps,
        train_initial_window=initial_train_window,
        max_eval_windows=3,
        timeout_minutes=opt_timeout_minutes,
    )
    study.optimize(objective, n_trials=opt_trials, timeout=(None if opt_timeout_minutes is None else int(opt_timeout_minutes * 60)))

    best_params = study.best_trial.params
    # Add default params that might be needed downstream
    for key in ["batch_size", "optimizer", "learning_rate"]:
        if key not in best_params:
            best_params[key] = 64 if key == "batch_size" else ("adam" if key == "optimizer" else 1e-3)

    # Save study summary
    os.makedirs(results_dir, exist_ok=True)
    study_path = os.path.join(results_dir, "optuna_study_summary.json")
    with open(study_path, "w") as f:
        json.dump({"best_value": study.best_value, "best_params": best_params}, f, indent=2)

    # 3) Train final model and evaluate
    # Create several rolling-origin splits for final evaluation:
    splits = list(rolling_origin_splits(T=data.shape[0], train_window=initial_train_window, fh=output_steps, step=output_steps))
    # Use up to 3 splits for evaluation (last ones)
    eval_splits = splits[-3:] if len(splits) >= 1 else [(slice(0, initial_train_window), slice(initial_train_window, initial_train_window + output_steps))]

    results = train_and_evaluate_full(
        data=data,
        input_steps=input_steps,
        output_steps=output_steps,
        best_params=best_params,
        train_end_idx=initial_train_window,
        rolling_eval_splits=eval_splits,
        save_dir=results_dir,
    )

    # Save final merged report
    final_report = {
        "best_params": best_params,
        "optuna_best_value": float(study.best_value),
        "evaluation": results,
    }
    final_report_path = os.path.join(results_dir, "final_report.json")
    with open(final_report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    return final_report


if __name__ == "__main__":
    # Quick defaults (reduce n_steps and trials for faster runs locally)
    FINAL_RESULTS = run_full_experiment(
        n_steps=2500,          # >= 2000 as requested
        n_series=6,            # >= 5 features
        input_steps=48,        # e.g., use 48-step lookback
        output_steps=24,       # forecast horizon (multi-step)
        initial_train_window=1800,
        opt_trials=12,         # number of Optuna trials (increase for better tuning)
        opt_timeout_minutes=None,
        results_dir="./results",
    )
