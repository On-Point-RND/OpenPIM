"""Unified experiment metrics + spectra recording."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


RUN_METRICS_NAME = "run_metrics.csv"
SPECTRA_NAME = "spectra.npz"


class ExperimentRecorder:
    """
    Collects per-log-step metrics and PSDs; writes:
      - run_metrics.csv  (one row per iteration × channel)
      - spectra.npz      (freqs + psd arrays over log steps)
    """

    def __init__(self, path_dir_save: str, seed: int = 0):
        self.path_dir_save = path_dir_save
        self.seed = seed
        self.rows: List[dict] = []
        self.iterations: List[int] = []
        self.freqs: Optional[np.ndarray] = None
        self._psd: Dict[str, List[np.ndarray]] = {
            "rx": [],
            "pred": [],
            "err": [],
            "noise": [],
        }
        os.makedirs(path_dir_save, exist_ok=True)

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.path_dir_save, RUN_METRICS_NAME)

    @property
    def spectra_path(self) -> str:
        return os.path.join(self.path_dir_save, SPECTRA_NAME)

    def log_step(
        self,
        *,
        iteration: int,
        time_min: float,
        lr: float,
        train_loss: float,
        test_loss: float,
        nmse_by_ch: Dict[str, float],
        reduction_by_ch: Dict[str, float],
        powers: dict,
        powers_lite: dict,
        mean_reduction: float,
        mean_res_lite: float,
        freqs: np.ndarray,
        psd_rx: np.ndarray,
        psd_pred: np.ndarray,
        psd_err: np.ndarray,
        psd_noise: np.ndarray,
    ) -> None:
        n_channels = len(powers["gt"])
        for ch in range(n_channels):
            self.rows.append(
                {
                    "iteration": iteration,
                    "channel": ch,
                    "time_min": time_min,
                    "lr": lr,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "nmse": float(nmse_by_ch.get(f"CH_{ch}", np.nan)),
                    "reduction_level": float(
                        reduction_by_ch.get(f"CH_{ch}", np.nan)
                    ),
                    "rxa_db": powers["gt"][ch] - powers["noise"][ch],
                    "err_db": powers["err"][ch] - powers["noise"][ch],
                    "noise_db": powers["noise"][ch],
                    "rxa_lite_db": powers_lite["gt"][ch],
                    "res_lite_db": powers_lite["err"][ch],
                    "mean_reduction": mean_reduction,
                    "mean_res_lite": mean_res_lite,
                }
            )

        if self.freqs is None:
            self.freqs = np.asarray(freqs)
        self.iterations.append(iteration)
        self._psd["rx"].append(np.asarray(psd_rx))
        self._psd["pred"].append(np.asarray(psd_pred))
        self._psd["err"].append(np.asarray(psd_err))
        self._psd["noise"].append(np.asarray(psd_noise))

        self.flush_metrics()

    def flush_metrics(self) -> None:
        if not self.rows:
            return
        pd.DataFrame(self.rows).to_csv(self.metrics_path, index=False)

    def save_spectra(self, meta: dict | None = None) -> None:
        if not self.iterations or self.freqs is None:
            return
        payload = {
            "iterations": np.asarray(self.iterations, dtype=np.int64),
            "freqs": self.freqs,
            "psd_rx": np.stack(self._psd["rx"], axis=0),
            "psd_pred": np.stack(self._psd["pred"], axis=0),
            "psd_err": np.stack(self._psd["err"], axis=0),
            "psd_noise": np.stack(self._psd["noise"], axis=0),
        }
        if meta:
            for k, v in meta.items():
                payload[k] = np.asarray(v)
        np.savez_compressed(self.spectra_path, **payload)
