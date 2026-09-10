"""Unified experiment metrics + spectra recording."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


RUN_METRICS_NAME = "run_metrics.csv"
SPECTRA_NAME = "spectra.npz"
BARplot_POWERS_NAME = "barplot_powers.npz"


class ExperimentRecorder:
    """
    Collects per-log-step metrics and PSDs; writes:
      - run_metrics.csv  (one row per iteration × channel)
      - spectra.npz      (freqs + psd arrays over log steps)
    """

    def __init__(
        self,
        path_dir_save: str,
        seed: int = 0,
        append_metrics: bool = False,
    ):
        self.path_dir_save = path_dir_save
        self.seed = seed
        self.append_metrics = append_metrics
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

    @property
    def barplot_powers_path(self) -> str:
        return os.path.join(self.path_dir_save, BARplot_POWERS_NAME)

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
        freqs: Optional[np.ndarray] = None,
        psd_rx: Optional[np.ndarray] = None,
        psd_pred: Optional[np.ndarray] = None,
        psd_err: Optional[np.ndarray] = None,
        psd_noise: Optional[np.ndarray] = None,
    ) -> None:
        n_channels = len(powers["gt"])
        for ch in range(n_channels):
            self.rows.append(
                {
                    "seed": self.seed,
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

        if freqs is not None and psd_rx is not None:
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
        new_df = pd.DataFrame(self.rows)
        if self.append_metrics and os.path.exists(self.metrics_path):
            old = pd.read_csv(self.metrics_path)
            if "seed" in old.columns:
                old = old[old["seed"] != self.seed]
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df.to_csv(self.metrics_path, index=False)

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

    def save_barplot_powers(
        self,
        powers: dict,
        powers_lite: dict,
        iteration: int,
    ) -> None:
        """Per-channel levels used by barplot_performance*.png (primary run only)."""
        n = len(powers["gt"])
        classic_rxa = np.asarray(
            [powers["gt"][i] - powers["noise"][i] for i in range(n)], dtype=np.float64
        )
        classic_err = np.asarray(
            [powers["err"][i] - powers["noise"][i] for i in range(n)], dtype=np.float64
        )
        np.savez_compressed(
            self.barplot_powers_path,
            iteration=np.int64(iteration),
            seed=np.int64(self.seed),
            classic_rxa_db=classic_rxa,
            classic_err_db=classic_err,
            lite_rxa_db=np.asarray(powers_lite["gt"], dtype=np.float64),
            lite_res_db=np.asarray(powers_lite["err"], dtype=np.float64),
        )
