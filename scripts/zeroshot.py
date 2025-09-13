import json
import os
import sys
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
os.environ["PYTHONPATH"] = str(root_dir)
sys.path.insert(0, str(root_dir))


def _find_exp(exp: str) -> str:
    ckpt_root = root_dir / "checkpoints"
    if not (ckpt_root / exp).exists():
        for subdir in ckpt_root.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(exp):
                print(f"| match ckpt by prefix: {subdir.name}")
                exp = subdir.name
                break
        else:
            raise click.BadParameter(
                f"There are no matching exp starting with '{exp}' in 'checkpoints' folder. "
                f"Please specify '--exp' as the folder name or prefix."
            )
    else:
        print(f"| found ckpt by name: {exp}")
    return exp


def _try_import_speechbrain():
    try:
        import speechbrain  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_speechbrain_installed():
    if _try_import_speechbrain():
        return True
    # Lazy install if not present; return False if installation fails
    import subprocess, sys as _sys
    try:
        print("| installing dependency: speechbrain")
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "speechbrain~=0.5.15"], stdout=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"| warn: speechbrain install failed, falling back to MFCC embeddings. ({e})")
        return False


def _load_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    import librosa
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype(np.float32), target_sr


def _embed_wav_speechbrain(wav_path: Path, device: str = "cpu") -> np.ndarray:
    ok = _ensure_speechbrain_installed()
    if not ok:
        raise RuntimeError("speechbrain not available")
    import torch
    from speechbrain.pretrained import EncoderClassifier

    # Load encoder (download weights if necessary)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=str(root_dir / ".cache" / "speechbrain_ecapa")
    )
    # Read audio
    wav, sr = _load_audio(wav_path, target_sr=16000)
    wav_t = torch.from_numpy(wav)[None]  # [1, T]
    with torch.no_grad():
        emb = classifier.encode_batch(wav_t)
    emb = emb.squeeze(0).squeeze(0).cpu().numpy()  # [D]
    # Normalize to unit length
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm


def _embed_wav_content(wav_path: Path, device: str = "cpu", backend: str = "auto") -> np.ndarray:
    """Try to extract content features using torchaudio pipelines (HuBERT/Wav2Vec/WavLM),
    then aggregate to an utterance-level vector by mean+std pooling.
    """
    try:
        import torchaudio
        bundle = None
        backend = (backend or "auto").lower()
        # Explicit backend selection mapping
        prefer = []
        if backend == "hubert" or backend == "cvec":
            prefer = ["HUBERT_BASE", "WAV2VEC2_BASE", "WAV2VEC2_ASR_BASE_960H", "WAVLM_BASE"]
        elif backend == "wavlm":
            prefer = ["WAVLM_BASE", "HUBERT_BASE", "WAV2VEC2_BASE", "WAV2VEC2_ASR_BASE_960H"]
        elif backend == "wav2vec2":
            prefer = ["WAV2VEC2_BASE", "WAV2VEC2_ASR_BASE_960H", "HUBERT_BASE", "WAVLM_BASE"]
        else:  # auto
            prefer = ["HUBERT_BASE", "WAVLM_BASE", "WAV2VEC2_BASE", "WAV2VEC2_ASR_BASE_960H"]
        for name in prefer:
            if hasattr(torchaudio.pipelines, name):
                bundle = getattr(torchaudio.pipelines, name)
                break
        if bundle is None:
            raise ImportError("No suitable torchaudio pipeline found")
        model = bundle.get_model().to(device).eval()
        sr_target = getattr(bundle, "sample_rate", 16000)
        wav, _ = _load_audio(wav_path, target_sr=sr_target)
        wav_t = torch.from_numpy(wav)[None].to(device)
        with torch.no_grad():
            if hasattr(model, "extract_features"):
                out = model.extract_features(wav_t)
                # torchaudio returns (features, length) or list; handle both
                if isinstance(out, tuple):
                    features = out[0]
                else:
                    features = out
                if isinstance(features, (list, tuple)):
                    feat = features[-1].squeeze(0).cpu().numpy()
                else:
                    feat = features.squeeze(0).cpu().numpy()
            else:
                # Fallback: direct forward (may return emissions)
                y = model(wav_t)
                feat = y.squeeze(0).detach().cpu().numpy()
        # Aggregate
        if feat.ndim == 1:
            vec = feat
        else:
            mu = feat.mean(axis=0)
            sd = feat.std(axis=0)
            vec = np.concatenate([mu, sd], axis=0)
        vec = vec.astype(np.float32)
        # normalize to unit variance
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        return vec
    except Exception as e:
        raise RuntimeError(f"torchaudio content embedding unavailable: {e}")


def _embed_wav_mfcc(wav_path: Path, device: str = "cpu") -> np.ndarray:
    """Fallback embedding: average MFCCs (no network dependency)."""
    import librosa
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    # Pre-emphasis and trim
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < sr // 2:
        # pad short clips
        y = np.pad(y, (0, sr - len(y)), mode="reflect")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=512, hop_length=160, n_mels=64)
    # delta features
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, d1, d2], axis=0)  # [120, T]
    emb = feat.mean(axis=1)
    emb = emb - emb.mean()
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.astype(np.float32)


def _embed_wav(wav_path: Path, device: str = "cpu", backend: str = "auto") -> np.ndarray:
    # Prefer content features (HuBERT/WavLM/Wav2Vec2), else MFCC
    if (backend or "auto").lower() == "mfcc":
        return _embed_wav_mfcc(wav_path, device=device)
    try:
        return _embed_wav_content(wav_path, device=device, backend=backend)
    except Exception as e_content:
        print(f"| info: content embed failed: {e_content}")
        try:
            return _embed_wav_mfcc(wav_path, device=device)
        except Exception as e_mfcc:
            print(f"| warn: MFCC embed failed: {e_mfcc}")
            raise


def _project_to_hidden(vec: np.ndarray, hidden_size: int, embed_weight: torch.Tensor | None = None) -> np.ndarray:
    """Project arbitrary-dim vec to model hidden size.
    - Normalize vec to zero-mean unit-std.
    - If dim mismatch, apply a fixed random projection to H dims.
    - If embed_weight provided (num_spk, H), re-scale per-dim to match its mean/std.
    """
    v = vec.astype(np.float32)
    v = (v - v.mean()) / (v.std() + 1e-6)
    H = int(hidden_size)
    if v.shape[0] != H:
        rng = np.random.RandomState(20240904)
        W = rng.normal(0.0, 1.0 / max(1.0, np.sqrt(v.shape[0])), size=(v.shape[0], H)).astype(np.float32)
        v = v @ W
    if embed_weight is not None:
        w = embed_weight.detach().cpu().numpy()
        mu = w.mean(axis=0)
        sd = w.std(axis=0) + 1e-6
        v = mu + v * sd
    return v.astype(np.float32)


def _gather_speaker_prototypes(spk_map_path: Path) -> List[Tuple[str, Path]]:
    """Return a list of (spk_name, wav_path) prototypes found in samples/.

    We search for files like '*_<spk_name>_*.wav' or '*<spk_name>*.wav' under 'samples/'.
    """
    samples_dir = root_dir / "samples"
    if not samples_dir.exists():
        return []

    with open(spk_map_path, "r", encoding="utf-8") as f:
        spk_map = json.load(f)
    # Normalize to a set of speaker names. Prefer keys if values are ints.
    if all(isinstance(v, int) for v in spk_map.values()):
        spk_names = list(spk_map.keys())
    else:
        spk_names = list(spk_map.values())

    prototypes: List[Tuple[str, Path]] = []
    for spk in spk_names:
        # Prefer patterns with '_<spk>_' in filename
        candidates = list(samples_dir.glob(f"*_{spk}_*.wav"))
        if not candidates:
            candidates = list(samples_dir.glob(f"*{spk}*.wav"))
        if candidates:
            # Take the shortest file to speed up embedding and avoid long songs
            candidates.sort(key=lambda p: p.stat().st_size)
            prototypes.append((spk, candidates[0]))
    return prototypes


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _softmax(x: np.ndarray, temp: float = 0.07) -> np.ndarray:
    x = x / max(temp, 1e-6)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / (s + 1e-9)


def _auto_pick_reference_wav(pref_dirnames: Tuple[str, ...] = ("测试用音色", "參考音色", "reference", "ref", "音色")) -> Path:
    # try exact folder names first
    for name in pref_dirnames:
        p = root_dir / name
        if p.exists() and p.is_dir():
            wavs = list(p.glob("*.wav"))
            if wavs:
                return wavs[0]
    # fuzzy search any dir that contains '音色'
    for child in root_dir.iterdir():
        if child.is_dir() and ("色" in child.name or "音色" in child.name):
            wavs = list(child.glob("*.wav"))
            if wavs:
                return wavs[0]
    # fallback to samples
    samples = list((root_dir / "samples").glob("*.wav"))
    if samples:
        return samples[0]
    raise FileNotFoundError("No reference wav found. Please pass --ref explicitly.")


@click.command(help="Zero-shot timbre mix inference for DiffSinger acoustic model")
@click.argument(
    "proj",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path, resolve_path=True),
    metavar="DS_FILE",
)
@click.option("--exp", type=str, required=True, callback=lambda ctx, p, v: _find_exp(v), help="Model exp folder or prefix")
@click.option("--ckpt", type=click.IntRange(min=0), required=False, help="Checkpoint steps")
@click.option("--ref", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=False,
              help="Reference dry voice wav")
@click.option("--lang", type=click.STRING, required=False, help="Default language (e.g. zh, ja)")
@click.option("--out", type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path), required=False,
              help="Output folder")
@click.option("--title", type=click.STRING, required=False, help="Output title (filename)")
@click.option("--seed", type=click.INT, default=-1, show_default=True, help="Random seed")
@click.option("--steps", type=click.IntRange(min=1), required=False, help="Diffusion sampling steps override")
@click.option("--backend", type=click.Choice(["auto", "hubert", "cvec", "wavlm", "wav2vec2", "mfcc"], case_sensitive=False),
              default="auto", show_default=True, help="Embedding backend for zero-shot timbre")
def main(proj: Path, exp: str, ckpt: int, ref: Path, lang: str, out: Path, title: str,
         seed: int, steps: int, backend: str):
    # Load DS params
    with open(proj, "r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, list):
        params = [params]
    if len(params) == 0:
        print("The input file is empty.")
        sys.exit(1)

    # Prepare hparams
    sys.argv = [sys.argv[0], "--exp_name", exp, "--infer"]
    from utils.hparams import set_hparams, hparams
    set_hparams()
    if steps is not None:
        if hparams.get("use_shallow_diffusion", False):
            step_size = (1 - hparams.get("T_start_infer", 0.0)) / steps
            if "K_step_infer" in hparams:
                hparams["diff_speedup"] = round(step_size * hparams["K_step_infer"])
        else:
            if "timesteps" in hparams:
                hparams["diff_speedup"] = round(hparams["timesteps"] / steps)
        hparams["sampling_steps"] = steps

    # Resolve output
    name = proj.stem if not title else title
    if out is None:
        out = proj.parent / "zeroshot"
    out.mkdir(parents=True, exist_ok=True)

    # Ensure language field for multilingual models
    if (root_dir / "checkpoints" / exp / "lang_map.json").exists() and lang is None:
        # Default to zh if not provided
        lang = "zh"

    # Find reference wav if not given
    ref_wav = ref if ref is not None else _auto_pick_reference_wav()
    print(f"| reference wav: {ref_wav}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build inferencer early to access spk embed stats
    from inference.ds_acoustic import DiffSingerAcousticInfer
    infer_ins = DiffSingerAcousticInfer(load_vocoder=True, ckpt_steps=ckpt)
    print(f"| Model: {type(infer_ins.model)}")

    # Compute reference embedding and project to hidden size space
    raw_vec = _embed_wav(ref_wav, device=device, backend=backend)
    embed_weight = None
    try:
        embed_weight = infer_ins.model.fs2.spk_embed.weight
    except Exception:
        pass
    spk_vec = _project_to_hidden(raw_vec, hidden_size=int(hparams["hidden_size"]), embed_weight=embed_weight)

    # Patch params with a custom speaker embedding vector
    for param in params:
        if lang is not None:
            param["lang"] = lang
        param["spk_embed"] = spk_vec.tolist()
    try:
        infer_ins.run_inference(
            params,
            out_dir=out,
            title=f"{name}_zeroshot_{ref_wav.stem}",
            num_runs=1,
            seed=seed,
            save_mel=False,
        )
    except KeyboardInterrupt:
        sys.exit(-1)


if __name__ == "__main__":
    main()
