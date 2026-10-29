"""
Electrode-set spatial encoder for cross-subject iEEG-to-mel decoding.

Drop-in companion to 06_train_window_sweep_lstm.py. It replaces the
`X[:, :min_dim]` channel truncation with a permutation-invariant encoder that
accepts ALL electrodes of every subject (variable count), tags each electrode
with its anatomical position, and pools them into a fixed-size latent with
masked cross-attention (Perceiver-style). The existing LSTM head is kept.

Contents
  1. Electrode metadata loader (channels.tsv / electrodes.tsv, Destrieux labels)
  2. ROI masks (Broca-only, perisylvian speech network)  -> proposal 2
  3. High-gamma feature extraction on the continuous recording
  4. Padding utilities for variable electrode counts
  5. Keras model: per-electrode tokens + coordinate/region embedding
     + channel dropout + latent cross-attention + LSTM decoder  -> proposal 1
  6. Label-free reliability score (MC-dropout variance)          -> proposal 3

Assumptions to verify on the real data before use (marked VERIFY below):
  - the column order of nwbfile.acquisition["iEEG"].data matches the row
    order of *_channels.tsv;
  - the coordinate space of *_electrodes.tsv (check *_coordsystem.json).
    If coordinates are in native space, either warp them to MNI or set
    use_coords=False and rely on the atlas-region embedding only.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# 1. Electrode metadata
# ---------------------------------------------------------------------------

def load_electrode_meta(raw_dir: Path, subj: str) -> pd.DataFrame:
    """Return one row per recorded channel: name, region label, x, y, z."""
    ieeg_dir = Path(raw_dir) / subj / "ieeg"
    ch = pd.read_csv(sorted(glob.glob(str(ieeg_dir / "*channels.tsv")))[0], sep="\t")
    el_files = sorted(glob.glob(str(ieeg_dir / "*electrodes.tsv")))
    el = pd.read_csv(el_files[0], sep="\t") if el_files else pd.DataFrame(columns=["name", "x", "y", "z"])

    meta = ch[["name"]].copy()
    # iBIDS stores the Destrieux label in the `description` column of channels.tsv.
    meta["region"] = ch.get("description", pd.Series(["unknown"] * len(ch))).fillna("unknown").astype(str)
    meta = meta.merge(el[["name", "x", "y", "z"]], on="name", how="left")
    for c in ["x", "y", "z"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta["has_coord"] = meta[["x", "y", "z"]].notna().all(axis=1)
    meta[["x", "y", "z"]] = meta[["x", "y", "z"]].fillna(0.0)
    meta["shaft"] = meta["name"].str.extract(r"^([A-Za-z']+)")[0].fillna(meta["name"])
    return meta.reset_index(drop=True)  # VERIFY: order == NWB column order


def build_region_vocab(metas: list[pd.DataFrame]) -> dict[str, int]:
    """Region-label -> integer id. Built from TRAINING subjects only; id 0 = unknown/unseen."""
    labels = sorted({r for m in metas for r in m["region"].unique()} - {"unknown"})
    return {lab: i + 1 for i, lab in enumerate(labels)}


# ---------------------------------------------------------------------------
# 2. ROI definitions (Destrieux atlas label substrings)
# ---------------------------------------------------------------------------

BROCA_ROI = ["G_front_inf-Opercular", "G_front_inf-Triangul"]

SPEECH_NETWORK_ROI = BROCA_ROI + [
    "G_precentral", "G_postcentral", "S_central", "G_and_S_subcentral",   # vSMC
    "G_temp_sup", "G_temporal_transverse", "S_temporal_transverse",       # STG / Heschl
    "Lat_Fis", "G_insular_short", "G_Ins_lg_and_S_cent_ins", "S_circular_insula",
    "G_pariet_inf-Supramar",                                              # SMG
]


def roi_mask(meta: pd.DataFrame, roi: list[str] | None) -> np.ndarray:
    """Boolean keep-mask over channels; roi=None keeps all channels."""
    if roi is None:
        return np.ones(len(meta), dtype=bool)
    return meta["region"].apply(lambda r: any(k in r for k in roi)).to_numpy()


def roi_coverage_table(metas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-subject contact counts per ROI. Report this before any ROI experiment."""
    rows = []
    for subj, m in metas.items():
        rows.append({
            "subject": subj,
            "all": len(m),
            "broca": int(roi_mask(m, BROCA_ROI).sum()),
            "speech_network": int(roi_mask(m, SPEECH_NETWORK_ROI).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Feature extraction on the CONTINUOUS recording (before epoching)
# ---------------------------------------------------------------------------

def highgamma_features(ieeg: np.ndarray, fs: float, meta: pd.DataFrame | None = None,
                       frame_rate: float = 100.0, win_sec: float = 0.05,
                       band=(70.0, 170.0), line_hz: float = 50.0) -> np.ndarray:
    """Log high-gamma envelope per channel at `frame_rate`, shape [frames, C].

    Filtering is done once on the full recording, so short word events no longer
    suffer filtfilt edge effects (a likely reason the bandpower ablation underperformed).
    Re-reference: common average within each shaft if `meta` is given, else global CAR.
    """
    x = ieeg.astype(np.float64)
    if meta is not None:
        for _, idx in meta.groupby("shaft").indices.items():
            x[:, idx] -= x[:, idx].mean(axis=1, keepdims=True)
    else:
        x -= x.mean(axis=1, keepdims=True)

    for h in (1, 2, 3):  # 50 Hz line noise and harmonics
        f0 = line_hz * h
        sos = butter(4, [(f0 - 2) / (fs / 2), (f0 + 2) / (fs / 2)], btype="bandstop", output="sos")
        x = sosfiltfilt(sos, x, axis=0)
    sos = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="bandpass", output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, x, axis=0), axis=0))

    hop, win = int(round(fs / frame_rate)), int(round(win_sec * fs))
    n = 1 + (len(env) - win) // hop
    out = np.stack([env[i * hop:i * hop + win].mean(axis=0) for i in range(n)])
    return np.log(out + 1e-8).astype(np.float32)


def event_windows(feat: np.ndarray, frame_times: np.ndarray, target_times: np.ndarray,
                  lag: int, ctx: int) -> np.ndarray:
    """Context windows taken from the CONTINUOUS feature stream.

    Unlike windowing over concatenated event frames, the lagged context for the
    first frames of a word comes from the real pre-onset activity of that trial,
    not from the tail of the previous word.
    """
    idx = np.searchsorted(frame_times, target_times)
    starts = idx - lag - ctx
    valid = starts >= 0
    out = np.stack([feat[s:s + ctx] for s in starts[valid]])
    return out, valid


# ---------------------------------------------------------------------------
# 4. Padding for variable electrode counts
# ---------------------------------------------------------------------------

def pad_electrodes(X: np.ndarray, meta: pd.DataFrame, region_vocab: dict[str, int],
                   c_max: int, keep: np.ndarray | None = None):
    """X: [N, T, C, F] for one subject -> padded arrays with an electrode mask.

    Returns feats [N,T,c_max,F], coords [N,c_max,3], region [N,c_max], mask [N,c_max].
    """
    if keep is not None:
        X, meta = X[:, :, keep], meta[keep].reset_index(drop=True)
    n, t, c, f = X.shape
    if c > c_max:
        raise ValueError(f"{c} channels > c_max={c_max}")
    feats = np.zeros((n, t, c_max, f), np.float32)
    feats[:, :, :c] = X
    coords = np.zeros((c_max, 3), np.float32)
    coords[:c] = meta[["x", "y", "z"]].to_numpy(np.float32)
    region = np.zeros(c_max, np.int32)
    region[:c] = [region_vocab.get(r, 0) for r in meta["region"]]
    mask = np.zeros(c_max, np.float32)
    mask[:c] = 1.0
    rep = lambda a: np.broadcast_to(a, (n,) + a.shape).copy()
    return feats, rep(coords), rep(region), rep(mask)


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package="spatial")
class FourierCoords(layers.Layer):
    """Sinusoidal embedding of (x, y, z) in mm, with optional train-time jitter."""

    def __init__(self, n_freqs: int = 6, jitter_mm: float = 4.0, **kw):
        super().__init__(**kw)
        self.n_freqs, self.jitter_mm = n_freqs, jitter_mm

    def call(self, coords, training=None):
        if training and self.jitter_mm > 0:
            coords = coords + tf.random.normal(tf.shape(coords), stddev=self.jitter_mm)
        freqs = (2.0 ** tf.range(self.n_freqs, dtype=tf.float32)) * (np.pi / 100.0)
        ang = coords[..., None] * freqs                         # [B,C,3,K]
        emb = tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)   # [B,C,3,2K]
        return tf.reshape(emb, tf.concat([tf.shape(coords)[:2], [6 * self.n_freqs]], 0))

    def get_config(self):
        return {**super().get_config(), "n_freqs": self.n_freqs, "jitter_mm": self.jitter_mm}


@keras.utils.register_keras_serializable(package="spatial")
class ChannelDropout(layers.Layer):
    """Randomly removes whole electrodes during training (simulates coverage shift)."""

    def __init__(self, rate: float = 0.25, **kw):
        super().__init__(**kw)
        self.rate = rate

    def call(self, mask, training=None):
        if not training or self.rate <= 0:
            return mask
        keep = tf.cast(tf.random.uniform(tf.shape(mask)) > self.rate, mask.dtype)
        dropped = mask * keep
        # never drop every electrode of a sample
        none_left = tf.reduce_sum(dropped, axis=1, keepdims=True) < 1.0
        return tf.where(none_left, mask, dropped)

    def get_config(self):
        return {**super().get_config(), "rate": self.rate}


@keras.utils.register_keras_serializable(package="spatial")
class LatentCrossAttention(layers.Layer):
    """Per-time-step masked cross-attention from L learned latents to C electrode tokens."""

    def __init__(self, n_latents: int = 8, d_model: int = 32, n_heads: int = 4, **kw):
        super().__init__(**kw)
        self.n_latents, self.d_model, self.n_heads = n_latents, d_model, n_heads
        self.mha = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.norm = layers.LayerNormalization()

    def build(self, input_shape):
        self.latents = self.add_weight(name="latents", shape=(self.n_latents, self.d_model),
                                       initializer="glorot_uniform")
        super().build(input_shape)

    def call(self, tokens, mask):
        # tokens [B,T,C,d], mask [B,C]
        b, t, c = tf.shape(tokens)[0], tf.shape(tokens)[1], tf.shape(tokens)[2]
        flat = tf.reshape(tokens, (b * t, c, self.d_model))
        m = tf.repeat(mask, repeats=t, axis=0)                         # [B*T, C]
        attn_mask = tf.tile(tf.cast(m, tf.bool)[:, None, :], (1, self.n_latents, 1))
        q = tf.tile(self.latents[None], (b * t, 1, 1))
        z = self.norm(q + self.mha(q, flat, flat, attention_mask=attn_mask))
        return tf.reshape(z, (b, t, self.n_latents * self.d_model))

    def get_config(self):
        return {**super().get_config(), "n_latents": self.n_latents,
                "d_model": self.d_model, "n_heads": self.n_heads}


def build_spatial_lstm(t_steps: int, c_max: int, n_feat: int, n_regions: int, n_mels: int = 23,
                       d_model: int = 32, n_latents: int = 8, channel_drop: float = 0.25,
                       use_coords: bool = True, use_regions: bool = True) -> keras.Model:
    feats = keras.Input((t_steps, c_max, n_feat), name="feats")
    coords = keras.Input((c_max, 3), name="coords")
    region = keras.Input((c_max,), dtype="int32", name="region")
    mask = keras.Input((c_max,), name="mask")

    # shared per-electrode temporal value embedding (same weights for every electrode)
    tok = layers.Dense(d_model, activation="gelu", name="value_embed")(feats)       # [B,T,C,d]

    spatial = []
    if use_coords:
        spatial.append(layers.Dense(d_model, name="coord_proj")(FourierCoords(name="fourier")(coords)))
    if use_regions:
        spatial.append(layers.Embedding(n_regions + 1, d_model, name="region_embed")(region))
    if spatial:
        pos = spatial[0] if len(spatial) == 1 else layers.Add()(spatial)            # [B,C,d]
        tok = layers.Lambda(lambda a: a[0] + a[1][:, None])([tok, pos])

    m = ChannelDropout(channel_drop, name="channel_dropout")(mask)
    z = LatentCrossAttention(n_latents, d_model, name="latent_xattn")(tok, m)        # [B,T,L*d]

    # unchanged decoder head from the paper
    h = layers.LSTM(64, dropout=0.2)(z)
    h = layers.Dropout(0.25)(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    out = layers.Dense(n_mels, name="mel")(h)

    model = keras.Model([feats, coords, region, mask], out, name="spatial_lstm")
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="mse")
    return model


# ---------------------------------------------------------------------------
# 6. Label-free reliability (no ground-truth audio needed at test time)
# ---------------------------------------------------------------------------

def mc_dropout_uncertainty(model: keras.Model, inputs, n_passes: int = 20, batch: int = 512):
    """Mean prediction and per-frame predictive variance with dropout kept on."""
    preds = []
    for _ in range(n_passes):
        p = [model(tuple(a[i:i + batch] for a in inputs), training=True).numpy()
             for i in range(0, len(inputs[0]), batch)]
        preds.append(np.concatenate(p))
    preds = np.stack(preds)                        # [P, N, M]
    return preds.mean(0), preds.var(0).mean(-1)    # frame score: higher = less reliable


# ---------------------------------------------------------------------------
# Smoke test with random data (two "subjects" with different electrode counts)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, F, C_MAX = 20, 1, 128
    batches = []
    for c in (54, 117):
        meta = pd.DataFrame({"name": [f"A{i}" for i in range(c)],
                             "region": rng.choice(["G_precentral", "G_temp_sup-Lateral", "unknown"], c),
                             "x": rng.normal(0, 40, c), "y": rng.normal(0, 40, c), "z": rng.normal(0, 30, c)})
        vocab = {"G_precentral": 1, "G_temp_sup-Lateral": 2}
        batches.append(pad_electrodes(rng.normal(size=(64, T, c, F)).astype(np.float32), meta, vocab, C_MAX))
    inputs = tuple(np.concatenate(parts) for parts in zip(*batches))
    y = rng.normal(size=(len(inputs[0]), 23)).astype(np.float32)

    model = build_spatial_lstm(T, C_MAX, F, n_regions=2)
    model.summary()
    model.fit(inputs, y, epochs=1, batch_size=32, verbose=2)
    mean, unc = mc_dropout_uncertainty(model, inputs, n_passes=5)
    print("pred", mean.shape, "uncertainty", unc.shape)

    # permutation invariance check: shuffling electrode order must not change the output
    perm = rng.permutation(C_MAX)
    shuffled = (inputs[0][:, :, perm], inputs[1][:, perm], inputs[2][:, perm], inputs[3][:, perm])
    d = np.abs(model.predict(inputs, verbose=0) - model.predict(shuffled, verbose=0)).max()
    print("max |output change| after electrode shuffle:", float(d))
