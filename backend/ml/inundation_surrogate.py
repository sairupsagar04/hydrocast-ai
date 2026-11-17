# backend/ml/inundation_surrogate.py
"""
Model 2 — inundation (flood depth) prediction for HydroCast AI.

Two operating modes, selected automatically at startup:

1. **Trained surrogate** (preferred, when weights exist)
   If ``ml/model2_inundation.pth`` is present and loads, the
   FusionNeuralNetwork (U-Net style) runs exactly as before and its
   output grid is returned. ``model.weights_loaded`` is True and
   ``model.last_prediction_source`` is ``"trained_surrogate"``.

2. **Bathtub physics fallback** (when weights are missing)
   The weights file is NOT in the repository, so historically the
   network ran on random initialised weights and main.py consumed
   arbitrary numbers as flood depth. That is no longer possible: with
   no trained weights, ``predict_depth_grid`` uses a deterministic,
   volume-conserving "bathtub" inundation model on the real DEM from
   SpatialEngine:

       runoff volume  =  cumulative rainfall (m) x runoff coefficient
                         x window area
       water level L  =  the level for which the ponded volume in all
                         cells below L exactly equals the runoff volume
       depth(cell)    =  max(0, L - elevation(cell))

   Properties by construction: zero rain -> zero depth everywhere;
   more rain -> a strictly higher water level; the lowest ground
   floods first and deepest; total ponded volume is conserved.
   ``model.last_prediction_source`` is ``"physics_bathtub"`` and
   ``model.last_water_level_m`` holds the solved water level.

   The bathtub model is a screening-level estimator: it assumes water
   redistributes instantly across the window and is fully connected,
   and it does not remove water through the drainage network (main.py
   layers its per-road conduit/manhole logic on top). Training the
   surrogate on SWMM-generated runs remains the long-term path — when
   those weights are dropped in, mode 1 takes over automatically.

Units:
- ``feature_stack`` is the SpatialEngine stack: channel 0 is DEM
  elevation in metres, channel 2 is the roughness coefficient.
- ``rain_grid`` follows main.py's convention: rain RATE in mm/hr.
- ``cumulative_rain_mm`` is the event accumulation in mm. If a caller
  does not supply it, it is approximated as mean(rain_grid) x 1 hour.
- Returned depth grids are in METRES in both modes.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Bathtub inundation model (pure NumPy, volume-conserving)
# ---------------------------------------------------------------------------


def bathtub_depth_grid(
    dem_grid: np.ndarray,
    cumulative_rain_mm: float,
    runoff_coefficient: float,
) -> Tuple[np.ndarray, float]:
    """
    Solve the bathtub water level for a DEM and a runoff volume.

    Returns (depth_grid_m, water_level_m). ``depth_grid_m`` has the same
    shape as ``dem_grid``; the mean of the returned depths equals
    ``cumulative_rain_mm / 1000 * runoff_coefficient`` exactly (up to
    floating point), i.e. the ponded volume equals the runoff volume
    for any uniform cell size, because cell area cancels.

    Method: sort all elevations. For a candidate water level equal to
    the k-th smallest elevation, the ponded mean depth is
    ``g(k) = (elev_k * k - sum of k smallest elevations) / N``, which is
    non-decreasing in k. Find the largest k with ``g(k) <= target`` and
    solve the level analytically inside that segment:
    ``L = (target * N + sum of k smallest elevations) / k``.
    """
    dem = np.asarray(dem_grid, dtype=np.float64)
    if dem.size == 0:
        return np.zeros_like(dem, dtype=np.float32), 0.0

    runoff_coefficient = float(np.clip(runoff_coefficient, 0.0, 1.0))
    target_mean_depth_m = (float(cumulative_rain_mm) / 1000.0) * runoff_coefficient

    if target_mean_depth_m <= 0.0:
        return np.zeros(dem.shape, dtype=np.float32), float(dem.min())

    flat = np.sort(dem.ravel())
    n = flat.size
    prefix = np.concatenate(([0.0], np.cumsum(flat)))  # prefix[k] = sum of k smallest
    ks = np.arange(1, n + 1, dtype=np.float64)
    # g(k) for k = 1..n, aligned so g[k-1] is the value for count k.
    g = (flat * ks - prefix[1:]) / n

    # Number of g values <= target == the segment count k*.
    k = int(np.searchsorted(g, target_mean_depth_m, side="right"))
    if k < 1:
        k = 1
    water_level = (target_mean_depth_m * n + prefix[k]) / k

    depth = np.clip(water_level - dem, 0.0, None)
    return depth.astype(np.float32), float(water_level)


# ---------------------------------------------------------------------------
# FusionNeuralNetwork — architecture unchanged so trained weights still load
# ---------------------------------------------------------------------------


class FusionNeuralNetwork(nn.Module):
    def __init__(self, weights_path: str = None):
        super(FusionNeuralNetwork, self).__init__()

        # Simple U-Net surrogate architecture (unchanged)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rain_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.weights_loaded: bool = False
        self.last_prediction_source: Optional[str] = None
        self.last_water_level_m: Optional[float] = None

        if weights_path and os.path.exists(weights_path):
            try:
                self.load_state_dict(
                    torch.load(weights_path, map_location=torch.device("cpu"))
                )
                self.weights_loaded = True
                print(f"[✓] Model 2 Inundation weights loaded from {weights_path}")
            except Exception as e:
                print(
                    f"[!] Warning: Failed loading model weights ({e}). "
                    "Trained surrogate unavailable — bathtub physics fallback will be used."
                )
        else:
            print(
                "[!] Notice: No Model 2 weights found. The surrogate will NOT "
                "run untrained: predict_depth_grid uses the deterministic "
                "bathtub physics model on the real DEM instead."
            )

        self.eval()

    def forward(self, terrain_stack: torch.Tensor, rain_tensor: torch.Tensor) -> torch.Tensor:
        feat = self.encoder_conv(terrain_stack)
        rain_feat = self.rain_conv(rain_tensor)
        combined = torch.cat([feat, rain_feat], dim=1)
        out = self.decoder_conv(combined)
        return out

    # ------------------------------------------------------------------
    # Prediction dispatch
    # ------------------------------------------------------------------

    def predict_depth_grid(
        self,
        rain_grid: np.ndarray,
        feature_stack: np.ndarray,
        cumulative_rain_mm: Optional[float] = None,
        runoff_coefficient: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return a flood-depth grid in METRES with the same HxW as the
        feature stack. Uses the trained surrogate when weights are
        loaded, otherwise the bathtub physics model.

        ``rain_grid`` is rain rate in mm/hr (main.py convention) and is
        only used directly by the trained surrogate, or to approximate
        ``cumulative_rain_mm`` when the caller does not supply it.
        """
        if self.weights_loaded:
            return self._predict_with_surrogate(rain_grid, feature_stack)
        return self._predict_with_physics(
            rain_grid, feature_stack, cumulative_rain_mm, runoff_coefficient
        )

    def _predict_with_surrogate(
        self, rain_grid: np.ndarray, feature_stack: np.ndarray
    ) -> np.ndarray:
        """Trained-network inference (original behaviour, cleaned up)."""
        self.eval()
        with torch.no_grad():
            inputs = torch.tensor(feature_stack, dtype=torch.float32).unsqueeze(0)
            rain_tensor = (
                torch.tensor(rain_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            output = self.forward(inputs, rain_tensor)
        self.last_prediction_source = "trained_surrogate"
        self.last_water_level_m = None
        return output.squeeze().cpu().numpy()

    def _predict_with_physics(
        self,
        rain_grid: np.ndarray,
        feature_stack: np.ndarray,
        cumulative_rain_mm: Optional[float],
        runoff_coefficient: Optional[float],
    ) -> np.ndarray:
        """Volume-conserving bathtub inundation on the real DEM."""
        stack = np.asarray(feature_stack, dtype=np.float32)
        if stack.ndim != 3 or stack.shape[0] < 1:
            raise ValueError(
                f"feature_stack must have shape (channels, H, W); got {stack.shape}"
            )
        dem_grid = stack[0]

        if cumulative_rain_mm is None:
            # rain_grid holds a rain RATE in mm/hr; approximate the event
            # accumulation as one hour at the mean rate. Callers that know
            # the real accumulation (main.py does, via sim_cm) should
            # pass it explicitly.
            cumulative_rain_mm = float(np.mean(rain_grid)) * 1.0

        if runoff_coefficient is None:
            # Infer from the roughness channel: urban concrete (0.015)
            # sheds almost everything; rural soil (0.140) absorbs most.
            # Coefficients mirror the ones main.py uses.
            roughness_mean = float(np.mean(stack[2])) if stack.shape[0] > 2 else 0.015
            runoff_coefficient = 0.85 if roughness_mean <= 0.05 else 0.18

        depth, water_level = bathtub_depth_grid(
            dem_grid, cumulative_rain_mm, runoff_coefficient
        )
        self.last_prediction_source = "physics_bathtub"
        self.last_water_level_m = water_level
        return depth
