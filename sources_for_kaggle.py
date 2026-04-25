import csv
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


@dataclass
class DataConfig:
    data_root: str = "asr-2026-spoken-numbers-recognition-challenge"
    sample_rate: int = 16_000
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class AugmentConfig:
    p_speed: float = 0.45
    speed_choices: List[float] = field(default_factory=lambda: [0.90, 0.95, 1.00, 1.05, 1.10])
    speed_probs: List[float] = field(default_factory=lambda: [0.12, 0.21, 0.34, 0.21, 0.12])
    p_noise: float = 0.65
    snr_min: float = 10.0
    snr_max: float = 32.0
    p_gain: float = 0.60
    gain_min_db: float = -6.0
    gain_max_db: float = 6.0
    p_shift: float = 0.45
    shift_max_s: float = 0.08
    p_drop_chunk: float = 0.20
    drop_chunk_max_s: float = 0.10
    p_filter: float = 0.16
    p_polarity: float = 0.10
    p_reverb: float = 0.16
    reverb_mix_min: float = 0.08
    reverb_mix_max: float = 0.20
    echo_delay_ms_min: float = 24.0
    echo_delay_ms_max: float = 58.0
    echo_decay_min: float = 0.18
    echo_decay_max: float = 0.38
    p_codec: float = 0.22
    codec_downsample_choices: List[int] = field(default_factory=lambda: [8000, 12000, 16000])
    codec_bits_choices: List[int] = field(default_factory=lambda: [6, 7, 8])
    codec_bandwidth_min_hz: float = 2800.0
    codec_bandwidth_max_hz: float = 7600.0


@dataclass
class ModelConfig:
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    d_model: int = 160
    encoder_layers: int = 7
    encoder_heads: int = 4
    encoder_ffn_dim: int = 384
    encoder_kernel: int = 31
    decoder_layers: int = 3
    decoder_heads: int = 4
    decoder_ffn_dim: int = 512
    dropout: float = 0.2
    max_target_len: int = 8
    spec_time_masks: int = 2
    spec_freq_masks: int = 2
    spec_time_width: int = 34
    spec_freq_width: int = 14


@dataclass
class TrainConfig:
    seed: int = 42
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-2
    epochs: int = 55
    grad_clip: float = 3.0
    label_smoothing: float = 0.05
    tf_start: float = 1.0
    tf_end: float = 0.45
    beam_size_val: int = 6
    beam_size_test: int = 7
    min_number: int = 1000
    max_number: int = 999999
    max_params: int = 5_000_000
    checkpoints_dir: str = None
    best_ckpt_name: str = "conformer_seq2seq_best.pt"
    last_ckpt_name: str = "conformer_seq2seq_last.pt"


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
DIGITS = [str(i) for i in range(10)]
VOCAB = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + DIGITS
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
PAD_ID = TOKEN2ID[PAD_TOKEN]
BOS_ID = TOKEN2ID[BOS_TOKEN]
EOS_ID = TOKEN2ID[EOS_TOKEN]
VOCAB_SIZE = len(VOCAB)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_number(text: str) -> List[int]:
    digits = [TOKEN2ID[c] for c in str(int(text))]
    return [BOS_ID] + digits + [EOS_ID]


def decode_tokens_to_number(token_ids: List[int], min_number: int, max_number: int) -> int:
    chars: List[str] = []
    for t in token_ids:
        if t == EOS_ID:
            break
        tok = ID2TOKEN.get(int(t), "")
        if tok in DIGITS:
            chars.append(tok)
    if not chars:
        return min_number
    value = int("".join(chars))
    return int(min(max(value, min_number), max_number))


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer_numbers(ref_num: int, hyp_num: int) -> float:
    ref = str(ref_num)
    hyp = str(hyp_num)
    return levenshtein(ref, hyp) / max(1, len(ref))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def good_resample_rate(orig_sr: int, speed: float, divisor: int = 100) -> int:
    raw = orig_sr / speed
    return max(divisor, int(round(raw / divisor) * divisor))


class WaveAugment:
    def __init__(self, cfg: AugmentConfig) -> None:
        self.cfg = cfg

    def _shift(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        max_shift = int(self.cfg.shift_max_s * sample_rate)
        if max_shift <= 0:
            return waveform
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return waveform
        return torch.roll(waveform, shifts=shift, dims=-1)

    def _drop_chunk(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        max_len = int(self.cfg.drop_chunk_max_s * sample_rate)
        if max_len <= 0 or waveform.numel() < 2:
            return waveform
        cut = random.randint(1, min(max_len, waveform.numel() - 1))
        start = random.randint(0, waveform.numel() - cut)
        out = waveform.clone()
        out[start: start + cut] = 0.0
        return out

    def _filter(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if random.random() < 0.5:
            cutoff = random.uniform(2500.0, 7000.0)
            return AF.lowpass_biquad(waveform, sample_rate, cutoff)
        cutoff = random.uniform(80.0, 420.0)
        return AF.highpass_biquad(waveform, sample_rate, cutoff)

    def _reverb_echo(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Lightweight synthetic room effect: early reflection + one delayed echo.
        mix = random.uniform(self.cfg.reverb_mix_min, self.cfg.reverb_mix_max)
        delay_ms = random.uniform(self.cfg.echo_delay_ms_min, self.cfg.echo_delay_ms_max)
        delay = max(1, int(sample_rate * delay_ms / 1000.0))
        decay = random.uniform(self.cfg.echo_decay_min, self.cfg.echo_decay_max)

        out = waveform.clone()
        if waveform.size(-1) > delay:
            out[..., delay:] = out[..., delay:] + decay * waveform[..., :-delay]

        early_delay = max(1, delay // 2)
        early_decay = 0.5 * decay
        if waveform.size(-1) > early_delay:
            out[..., early_delay:] = out[..., early_delay:] + early_decay * waveform[..., :-early_delay]

        return (1.0 - mix) * waveform + mix * out

    def _codec(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Codec-like degradation: bandwidth reduction + temporal resampling + quantization noise.
        ds_sr = random.choice(self.cfg.codec_downsample_choices)
        ds_sr = max(2000, min(int(ds_sr), sample_rate))
        out = waveform

        if ds_sr < sample_rate:
            out = AF.resample(out, sample_rate, ds_sr)
            out = AF.resample(out, ds_sr, sample_rate)

        nyquist = max(1000.0, sample_rate / 2.0 - 50.0)
        cutoff = random.uniform(self.cfg.codec_bandwidth_min_hz, min(self.cfg.codec_bandwidth_max_hz, nyquist))
        out = AF.lowpass_biquad(out, sample_rate, cutoff)

        bits = int(random.choice(self.cfg.codec_bits_choices))
        levels = float((1 << bits) - 1)
        out = torch.round((out.clamp(-1.0, 1.0) + 1.0) * 0.5 * levels) / levels
        out = out * 2.0 - 1.0
        return out

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        cfg = self.cfg
        if random.random() < cfg.p_speed and cfg.speed_choices:
            speed = random.choices(cfg.speed_choices, weights=cfg.speed_probs, k=1)[0]
            if abs(speed - 1.0) > 1e-6:
                pseudo_sr = good_resample_rate(sample_rate, speed, divisor=100)
                waveform = AF.resample(waveform, sample_rate, pseudo_sr)

        if random.random() < cfg.p_shift:
            waveform = self._shift(waveform, sample_rate)

        if random.random() < cfg.p_noise:
            snr_db = random.uniform(cfg.snr_min, cfg.snr_max)
            noise = torch.randn_like(waveform)
            sig_pow = waveform.pow(2).mean().clamp_min(1e-8)
            noise_pow = noise.pow(2).mean().clamp_min(1e-8)
            scale = torch.sqrt(sig_pow / (noise_pow * (10 ** (snr_db / 10.0))))
            waveform = waveform + scale * noise

        if random.random() < cfg.p_filter:
            waveform = self._filter(waveform, sample_rate)

        if random.random() < cfg.p_reverb:
            waveform = self._reverb_echo(waveform, sample_rate)

        if random.random() < cfg.p_codec:
            waveform = self._codec(waveform, sample_rate)

        if random.random() < cfg.p_drop_chunk:
            waveform = self._drop_chunk(waveform, sample_rate)

        if random.random() < cfg.p_gain:
            gain_db = random.uniform(cfg.gain_min_db, cfg.gain_max_db)
            waveform = waveform * (10 ** (gain_db / 20.0))

        if random.random() < cfg.p_polarity:
            waveform = -waveform

        return waveform.clamp(-1.0, 1.0), sample_rate


class SpokenNumbersDataset(Dataset):
    def __init__(
            self,
            csv_path: Path,
            data_root: Path,
            target_sr: int,
            is_train: bool,
            augment: Optional[WaveAugment] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.target_sr = target_sr
        self.is_train = is_train
        self.augment = augment
        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _get_resampler(self, src_sr: int) -> torchaudio.transforms.Resample:
        if src_sr not in self._resamplers:
            self._resamplers[src_sr] = torchaudio.transforms.Resample(src_sr, self.target_sr)
        return self._resamplers[src_sr]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        waveform, sr = torchaudio.load(str(self.data_root / row["filename"]))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = self._get_resampler(sr)(waveform)
            sr = self.target_sr

        if self.is_train and self.augment is not None:
            waveform, sr = self.augment(waveform, sr)

        return {
            "waveform": waveform.squeeze(0),
            "length": int(waveform.shape[-1]),
            "filename": row["filename"],
            "label_text": row.get("transcription") or None,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    waveforms = [x["waveform"] for x in batch]
    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    labels = [x["label_text"] for x in batch]

    has_labels = all(x is not None for x in labels)
    out = {
        "waveforms": pad_sequence(waveforms, batch_first=True),
        "lengths": lengths,
        "filenames": [x["filename"] for x in batch],
        "label_texts": labels,
    }
    if has_labels:
        token_tensors = [torch.tensor(encode_number(x), dtype=torch.long) for x in labels]
        out["targets"] = pad_sequence(token_tensors, batch_first=True, padding_value=PAD_ID)
    else:
        out["targets"] = None
    return out


def create_dataloaders(data_cfg: DataConfig, aug_cfg: AugmentConfig) -> Dict[str, DataLoader]:
    root = Path(data_cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    kwargs = {"num_workers": data_cfg.num_workers, "pin_memory": data_cfg.pin_memory, "collate_fn": collate_fn}
    if data_cfg.num_workers > 0:
        kwargs["persistent_workers"] = True

    return {
        "train": DataLoader(
            SpokenNumbersDataset(root / "train.csv", root, data_cfg.sample_rate, True, WaveAugment(aug_cfg)),
            batch_size=data_cfg.batch_size,
            shuffle=True,
            **kwargs,
        ),
        "dev": DataLoader(
            SpokenNumbersDataset(root / "dev.csv", root, data_cfg.sample_rate, False, None),
            batch_size=data_cfg.batch_size,
            shuffle=False,
            **kwargs,
        ),
        "test": DataLoader(
            SpokenNumbersDataset(root / "test.csv", root, data_cfg.sample_rate, False, None),
            batch_size=data_cfg.batch_size,
            shuffle=False,
            **kwargs,
        ),
    }


class NumberSeq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hop_length = cfg.hop_length

        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=cfg.n_fft,
                                                            hop_length=cfg.hop_length, n_mels=cfg.n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=cfg.spec_freq_width, iid_masks=True)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=cfg.spec_time_width, iid_masks=True)

        self.in_norm = nn.LayerNorm(cfg.n_mels)
        self.in_proj = nn.Linear(cfg.n_mels, cfg.d_model)

        self.encoder = torchaudio.models.Conformer(
            input_dim=cfg.d_model,
            num_heads=cfg.encoder_heads,
            ffn_dim=cfg.encoder_ffn_dim,
            num_layers=cfg.encoder_layers,
            depthwise_conv_kernel_size=cfg.encoder_kernel,
            dropout=cfg.dropout,
            use_group_norm=True,
            convolution_first=False,
        )

        self.token_emb = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_target_len, cfg.d_model)
        self.dec_drop = nn.Dropout(cfg.dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.decoder_heads,
            dim_feedforward=cfg.decoder_ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.decoder_layers)
        self.out_proj = nn.Linear(cfg.d_model, VOCAB_SIZE)

    def _apply_specaugment(self, feats: torch.Tensor) -> torch.Tensor:
        out = feats
        for _ in range(self.cfg.spec_freq_masks):
            out = self.freq_mask(out)
        for _ in range(self.cfg.spec_time_masks):
            out = self.time_mask(out)
        return out

    def _lengths_after_frontend(self, audio_lengths: torch.Tensor, max_t: int) -> torch.Tensor:
        return torch.clamp((audio_lengths // self.hop_length) + 1, min=1, max=max_t)

    def encode(self, waveforms: torch.Tensor, audio_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.to_db(self.melspec(waveforms))
        if self.training:
            feats = self._apply_specaugment(feats)

        x = self.in_proj(self.in_norm(feats.transpose(1, 2)))
        enc_lengths = self._lengths_after_frontend(audio_lengths, x.size(1))
        return self.encoder(x, enc_lengths)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def decode(self, memory: torch.Tensor, mem_lengths: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        bsz, tgt_len = tgt_tokens.shape
        pos = torch.arange(tgt_len, device=tgt_tokens.device).unsqueeze(0).expand(bsz, -1)
        tgt = self.dec_drop(self.token_emb(tgt_tokens) + self.pos_emb(pos))

        tgt_mask = self._causal_mask(tgt_len, tgt.device)
        tgt_key_padding_mask = tgt_tokens.eq(PAD_ID)
        mem_key_padding_mask = torch.arange(memory.size(1), device=memory.device).unsqueeze(0) >= mem_lengths.unsqueeze(
            1)

        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )
        return self.out_proj(dec_out)

    def forward(self, waveforms: torch.Tensor, audio_lengths: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        memory, mem_lengths = self.encode(waveforms, audio_lengths)
        return self.decode(memory, mem_lengths, tgt_tokens)

    @torch.no_grad()
    def predict_numbers_beam(
            self,
            waveforms: torch.Tensor,
            audio_lengths: torch.Tensor,
            beam_size: int,
            min_number: int,
            max_number: int,
            length_penalty: float = 0.6,
    ) -> Tuple[List[int], List[float]]:
        was_training = self.training
        self.eval()
        memory, mem_lengths = self.encode(waveforms, audio_lengths)

        results: List[int] = []
        scores: List[float] = []
        for b in range(memory.size(0)):
            mem_b = memory[b: b + 1]
            len_b = mem_lengths[b: b + 1]
            beams: List[Tuple[float, List[int], bool]] = [(0.0, [BOS_ID], False)]

            for _ in range(self.cfg.max_target_len - 1):
                new_beams: List[Tuple[float, List[int], bool]] = []
                for score, tokens, ended in beams:
                    if ended:
                        new_beams.append((score, tokens, True))
                        continue

                    tgt = torch.tensor(tokens, device=mem_b.device, dtype=torch.long).unsqueeze(0)
                    logits = self.decode(mem_b, len_b, tgt)[0, -1]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk = torch.topk(log_probs, k=min(beam_size, VOCAB_SIZE))

                    for val, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                        next_tokens = tokens + [int(idx)]
                        new_beams.append((score + float(val), next_tokens, int(idx) == EOS_ID))

                beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
                if all(e for _, _, e in beams):
                    break

            ranked: List[Tuple[float, int]] = []
            for score, tokens, _ in beams:
                core = tokens[1:]
                number = decode_tokens_to_number(core, min_number=min_number, max_number=max_number)
                n_tokens = max(1, len(core))
                norm_score = score / ((5.0 + n_tokens) / 6.0) ** length_penalty
                if number < min_number or number > max_number:
                    norm_score -= 4.0
                ranked.append((norm_score, number))

            ranked.sort(key=lambda x: x[0], reverse=True)
            scores.append(ranked[0][0])
            results.append(ranked[0][1])

        if was_training:
            self.train()
        return results, scores


def save_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        epoch: int,
        history: Dict[str, List[float]],
        best_val_cer: float,
        cfgs: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "history": history,
            "best_val_cer": float(best_val_cer),
            "cfgs": cfgs,
        },
        path,
    )


def load_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
) -> Tuple[int, Dict[str, List[float]], float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    history = ckpt.get("history", {"epoch": [], "train_loss": [], "val_loss": [], "val_cer": [], "lr": []})
    best_val_cer = float(ckpt.get("best_val_cer", float("inf")))
    # Fallback для старых/битых checkpoint: восстанавлием best из history.
    if not np.isfinite(best_val_cer):
        val_hist = history.get("val_cer", []) if isinstance(history, dict) else []
        if len(val_hist) > 0:
            best_val_cer = float(min(val_hist))

    return int(ckpt.get("epoch", 0)), history, best_val_cer


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.08):
    total_steps = max(2, int(total_steps))
    warmup_steps = min(max(20, int(total_steps * warmup_ratio)), total_steps - 1)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1),
                                                        eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def evaluate(
        model: NumberSeq2Seq,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        beam_size: int,
        min_number: int,
        max_number: int,
) -> Tuple[float, float]:
    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    cer_sum = 0.0
    cer_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            waveforms = batch["waveforms"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"]
            labels = batch["label_texts"]
            if targets is None:
                continue

            targets = targets.to(device)
            dec_in, dec_out = targets[:, :-1], targets[:, 1:]
            logits = model(waveforms, lengths, dec_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_out.reshape(-1))

            bsz = waveforms.size(0)
            val_loss_sum += float(loss.item()) * bsz
            val_count += bsz

            pred_numbers, _ = model.predict_numbers_beam(waveforms, lengths, beam_size, min_number, max_number)
            for t, p in zip([int(x) for x in labels], pred_numbers):
                cer_sum += cer_numbers(t, p)
                cer_count += 1

    return val_loss_sum / max(1, val_count), cer_sum / max(1, cer_count)


def plot_history(history: Dict[str, List[float]]) -> None:
    if not history.get("epoch"):
        print("History is empty.")
        return

    epochs = history["epoch"]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.grid(alpha=0.3)
    plt.xlabel("epoch")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_cer"], label="val_cer")
    plt.grid(alpha=0.3)
    plt.xlabel("epoch")
    plt.title("CER")
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_model(
        model: NumberSeq2Seq,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        start_epoch: int = 0,
        history: Optional[Dict[str, List[float]]] = None,
        best_val_cer: float = float("inf"),
        val_every_n_epochs: int = 3,
) -> Tuple[Dict[str, List[float]], float]:
    model.to(device)
    if history is None:
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_cer": [], "lr": []}

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=train_cfg.label_smoothing)
    ckpt_dir = Path(train_cfg.checkpoints_dir)
    best_ckpt = ckpt_dir / train_cfg.best_ckpt_name
    last_ckpt = ckpt_dir / train_cfg.last_ckpt_name

    def _last_finite(values: List[float], default: float) -> float:
        for v in reversed(values):
            if np.isfinite(v):
                return float(v)
        return float(default)

    last_val_loss = _last_finite(history.get("val_loss", []), float("nan"))
    last_val_cer = _last_finite(history.get("val_cer", []), float("inf"))

    for epoch in range(start_epoch + 1, train_cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        progress = (epoch - 1) / max(1, train_cfg.epochs - 1)
        tf_ratio = train_cfg.tf_start + (train_cfg.tf_end - train_cfg.tf_start) * progress

        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            waveforms = batch["waveforms"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"]
            if targets is None:
                continue

            targets = targets.to(device)
            dec_in = targets[:, :-1].clone()
            dec_out = targets[:, 1:].clone()

            if tf_ratio < 1.0:
                with torch.no_grad():
                    greedy_tokens = model(waveforms, lengths, dec_in).argmax(dim=-1)
                keep_teacher = torch.rand_like(dec_in.float()) < tf_ratio
                keep_teacher[:, 0] = True
                dec_in = torch.where(keep_teacher, dec_in, greedy_tokens)
                dec_in[:, 0] = BOS_ID

            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms, lengths, dec_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            bsz = waveforms.size(0)
            train_loss_sum += float(loss.item()) * bsz
            train_count += bsz

        train_loss = train_loss_sum / max(1, train_count)
        # Всегда валидируем первую эпоху, дальше — по интервалу и на последней эпохе.
        do_validate = (epoch == 1) or (epoch % max(1, val_every_n_epochs) == 0) or (epoch == train_cfg.epochs)
        if do_validate:
            val_loss, val_cer = evaluate(
                model, dev_loader, device, criterion, train_cfg.beam_size_val, train_cfg.min_number,
                train_cfg.max_number
            )
            last_val_loss, last_val_cer = float(val_loss), float(val_cer)
        else:
            val_loss, val_cer = last_val_loss, last_val_cer
        lr_now = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_cer"].append(val_cer)
        history["lr"].append(lr_now)

        cfgs = {"data_cfg": asdict(data_cfg), "model_cfg": asdict(model_cfg), "train_cfg": asdict(train_cfg)}
        if do_validate and val_cer < best_val_cer:
            best_val_cer = val_cer
            save_checkpoint(best_ckpt, model, optimizer, scheduler, epoch, history, best_val_cer, cfgs)
        save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch, history, best_val_cer, cfgs)

        if do_validate:
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_cer={val_cer:.4f} | best={best_val_cer:.4f} | lr={lr_now:.2e}"
            )
        else:
            best_str = f"{best_val_cer:.4f}" if np.isfinite(best_val_cer) else "inf"
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} | val=carried "
                f"(every {max(1, val_every_n_epochs)} epochs, val_loss={val_loss:.4f}, val_cer={val_cer:.4f}) "
                f"| best={best_str} | lr={lr_now:.2e}"
            )

    plot_history(history)
    return history, best_val_cer


def make_submission(
        model: NumberSeq2Seq,
        loader: DataLoader,
        output_csv: Path,
        device: torch.device,
        beam_size: int,
        min_number: int,
        max_number: int,
) -> Path:
    model.to(device)
    model.eval()
    filenames: List[str] = []
    preds: List[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Submission"):
            waveforms = batch["waveforms"].to(device)
            lengths = batch["lengths"].to(device)
            numbers, _ = model.predict_numbers_beam(waveforms, lengths, beam_size, min_number, max_number)
            filenames.extend(batch["filenames"])
            preds.extend([str(x) for x in numbers])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "transcription"])
        for fn, pr in zip(filenames, preds):
            writer.writerow([fn, pr])

    print(f"Saved submission: {output_csv}")
    return output_csv


def build_all(
        data_cfg: DataConfig,
        aug_cfg: AugmentConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        device: torch.device,
) -> Tuple[Dict[str, DataLoader], NumberSeq2Seq, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    loaders = create_dataloaders(data_cfg, aug_cfg)
    model = NumberSeq2Seq(model_cfg)

    n_params = count_parameters(model)
    print(f"Trainable params: {n_params:,}")
    if n_params > train_cfg.max_params:
        raise ValueError(f"Model has {n_params:,} params, expected <= {train_cfg.max_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=train_cfg.epochs * len(loaders["train"]))
    model.to(device)
    return loaders, model, optimizer, scheduler
