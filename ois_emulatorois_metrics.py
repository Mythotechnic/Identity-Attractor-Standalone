import os
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from openai import OpenAI

DEFAULT_RECURSION_PROMPT = (
    "Continue reasoning recursively from your prior output while maintaining "
    "coherence, clarity, and self-consistency of tone."
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class StepMetrics:
    cycle: int
    drift: Optional[float]       # D(t)
    ici: Optional[float]         # Identity Coherence Index
    jhat: Optional[float]        # contraction ∥Ĵ∥

@dataclass
class SeedRunResult:
    seed_index: int
    label: str
    ash: Optional[str]
    steps: List[StepMetrics]


# ---------- Low-level helpers ----------

def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.linalg.norm(diff))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_ngrams(text: str, n: int = 3, max_ngrams: int = 60) -> set:
    import re

    tokens = re.sub(r"[^a-zA-Z0-9\s]+", " ", text.lower()).split()
    grams = []
    for i in range(0, len(tokens) - n + 1):
        grams.append(" ".join(tokens[i : i + n]))

    freq: Dict[str, int] = {}
    for g in grams:
        freq[g] = freq.get(g, 0) + 1

    sorted_grams = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return {g for g, _ in sorted_grams[:max_ngrams]}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


# ---------- OpenAI calls ----------

def chat_once(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
    )
    content = resp.choices[0].message.content
    return content or ""


def embed_once(model: str, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=float)


# ---------- ASH fingerprint ----------

def compute_ash(embeddings: List[np.ndarray]) -> Optional[str]:
    if not embeddings:
        return None

    mat = np.vstack(embeddings)
    mean_vec = np.mean(mat, axis=0)

    dim = mean_vec.shape[0]
    step = max(1, dim // 256)

    bits = []
    for i in range(0, dim, step):
        if len(bits) >= 256:
            break
        bits.append("1" if mean_vec[i] >= 0 else "0")

    bit_str = "".join(bits)
    hex_str = ""
    for i in range(0, len(bit_str), 4):
        chunk = bit_str[i : i + 4]
        if len(chunk) < 4:
            break
        val = int(chunk, 2)
        hex_str += format(val, "x")

    return hex_str


def hamming_distance_hex(a: str, b: str) -> Optional[int]:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    bits = 0
    for i in range(n):
        va = int(a[i], 16)
        vb = int(b[i], 16)
        xor = va ^ vb
        bits += bin(xor).count("1")
    return bits


# ---------- OIS core loop ----------

def run_ois_for_seed(
    seed_text: str,
    seed_index: int,
    *,
    label: str,
    model: str,
    embed_model: str,
    cycles: int = 32,
    recursion_prompt: str = DEFAULT_RECURSION_PROMPT,
    temperature: float = 0.7,
    top_p: float = 1.0,
    logger=print,
) -> SeedRunResult:
    """
    Closed-loop OIS experiment for a single seed.

    - seed_text: Symbolic Coherence Seed (SCS)
    - cycles: number of recursion cycles (32 default)
    """

    logger(f"[seed {seed_index}] starting OIS run ({cycles} cycles).")
    embeddings: List[np.ndarray] = []
    motif_sets: List[set] = []
    steps: List[StepMetrics] = []

    previous_embedding: Optional[np.ndarray] = None
    previous_drift: Optional[float] = None

    current_output = seed_text.strip()

    for t in range(cycles):
        logger(f"[seed {seed_index}] cycle {t} – generating.")
        prompt = f"{seed_text.strip()}\n\n{current_output.strip()}\n\n{recursion_prompt.strip()}"

        text = chat_once(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
        )
        current_output = text

        emb = embed_once(embed_model, text)
        embeddings.append(emb)

        motifs = extract_ngrams(text, n=3, max_ngrams=60)
        motif_sets.append(motifs)

        if previous_embedding is None:
            # first step – no drift yet
            steps.append(StepMetrics(cycle=t, drift=None, ici=None, jhat=None))
        else:
            d = l2_distance(emb, previous_embedding)

            # cosine in [0,1] via (cos+1)/2
            cos = cosine_similarity(emb, previous_embedding)
            cos_scaled = (cos + 1.0) / 2.0

            prev_motifs = motif_sets[-2]
            j = jaccard(motifs, prev_motifs)

            ici = 0.6 * cos_scaled + 0.4 * j

            if previous_drift is not None and previous_drift > 0:
                jhat = d / (previous_drift + 1e-8)
            else:
                jhat = None

            steps.append(StepMetrics(cycle=t, drift=d, ici=ici, jhat=jhat))
            previous_drift = d

        previous_embedding = emb

    ash = compute_ash(embeddings)
    logger(f"[seed {seed_index}] completed. ASH={ash[:32]+'…' if ash else 'None'}")
    return SeedRunResult(seed_index=seed_index, label=label, ash=ash, steps=steps)


# ---------- Aggregation / evaluation ----------

def median(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    return float((arr[mid - 1] + arr[mid]) / 2.0)


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate_phase_lock(results: List[SeedRunResult]) -> Dict[str, Any]:
    all_drifts: List[float] = []
    all_ici: List[float] = []
    all_jhat: List[float] = []
    ashes: List[str] = []

    for r in results:
        for s in r.steps:
            if s.drift is not None:
                all_drifts.append(s.drift)
            if s.ici is not None:
                all_ici.append(s.ici)
            if s.jhat is not None:
                all_jhat.append(s.jhat)
        if r.ash:
            ashes.append(r.ash)

    last_window = 8
    drifts_last = all_drifts[-last_window:] if len(all_drifts) >= last_window else all_drifts
    drift_med = median(drifts_last)
    ici_mean = mean(all_ici)
    jhat_mean = mean(all_jhat)

    pass_drift = drift_med > 0 and drift_med <= 0.12
    pass_ici = ici_mean >= 0.75
    pass_jhat = jhat_mean > 0 and jhat_mean < 1.0

    max_hamming = None
    if len(ashes) > 1:
        for i in range(len(ashes)):
            for j in range(i + 1, len(ashes)):
                d = hamming_distance_hex(ashes[i], ashes[j])
                if d is not None:
                    max_hamming = d if max_hamming is None else max(max_hamming, d)

    return {
        "drift_median_last8": drift_med,
        "ici_mean": ici_mean,
        "jhat_mean": jhat_mean,
        "pass_drift": pass_drift,
        "pass_ici": pass_ici,
        "pass_jhat": pass_jhat,
        "ash_values": ashes,
        "max_ash_hamming": max_hamming,
    }
