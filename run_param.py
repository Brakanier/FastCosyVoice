#!/usr/bin/env python3
"""
Parameter sweep for LLM sampling parameters (temperature, top_p, top_k).

Generates the same text with different parameter combinations to evaluate
their effect on accent and voice quality. Uses offline (non-streaming) synthesis.
"""

import sys
import time
import os
import logging
import wave
from pathlib import Path
from itertools import product

sys.path.append('third_party/Matcha-TTS')

import torch
from fastcosyvoice import FastCosyVoice3

torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
REFERENCE_AUDIO = 'refs/019.wav'
OUTPUT_DIR = 'output/run_param'
INSTRUCTION = "You are a helpful assistant."

USE_TRT_FLOW = True
USE_TRT_LLM = True
TRT_LLM_DTYPE = 'bfloat16'
TRT_LLM_KV_CACHE_TOKENS = 8192

# LLM weights: 'llm.pt' (baseline) or 'llm.rl.pt' (RL-trained)
LLM_MODEL_NAME = 'llm.rl.pt'

SYNTHESIS_TEXT = (
    "И еще один дополнительный текст. Хочу проверить появится ли акцент в голосах "
    "при разных параметрах. А может надо писать немного другие предложения, например длинные, "
    "где есть запятые. Вау вот это да!"
)

# Parameter grid to sweep
TEMPERATURES = [0.7, 0.8, 1.0, 1.2, 1.4, 1.6]
TOP_P_VALUES = [0.95]
TOP_K_VALUES = [25]


def load_prompt_text(audio_path: str, instruction: str = INSTRUCTION) -> str:
    txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as f:
        transcription = f.read().strip()
    return f"{instruction}<|endofprompt|>{transcription}"


def synthesize_offline(
    cosyvoice: FastCosyVoice3,
    text: str,
    prompt_text: str,
    spk_id: str,
    sample_rate: int,
    output_path: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict:
    """Run offline (non-streaming) synthesis and return metrics."""
    start_time = time.time()
    audio_chunks: list[bytes] = []

    with torch.inference_mode():
        for pcm_bytes in cosyvoice.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            audio_chunks.append(pcm_bytes)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.time() - start_time

    if audio_chunks:
        full_pcm = b''.join(audio_chunks)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(full_pcm)
        audio_duration = len(full_pcm) / 2 / sample_rate
    else:
        audio_duration = 0.0

    rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

    return {
        'total_time': total_time,
        'audio_duration': audio_duration,
        'rtf': rtf,
    }


def main():
    print("=" * 70)
    print("Parameter Sweep: temperature / top_p / top_k")
    print("=" * 70)

    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}", exc_info=True)
        return

    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)

    # Build output subdirectory based on weights name
    weights_tag = LLM_MODEL_NAME.replace('.pt', '').replace('.', '_')
    out_dir = os.path.join(OUTPUT_DIR, weights_tag)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nReference audio: {REFERENCE_AUDIO}")
    print(f"LLM weights:    {LLM_MODEL_NAME}")
    print(f"Output dir:     {out_dir}")
    print(f"Text:           {SYNTHESIS_TEXT[:80]}...")

    combos = list(product(TEMPERATURES, TOP_P_VALUES, TOP_K_VALUES))
    print(f"Combinations:   {len(combos)}")

    # Load model
    print("\nLoading FastCosyVoice3...")
    load_start = time.time()

    cosyvoice = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_trt=USE_TRT_FLOW,
        load_trt_llm=USE_TRT_LLM,
        trt_llm_dtype=TRT_LLM_DTYPE,
        trt_llm_kv_cache_tokens=TRT_LLM_KV_CACHE_TOKENS,
        llm_model_name=LLM_MODEL_NAME,
    )

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f} sec")

    sample_rate = cosyvoice.sample_rate

    # Prepare speaker embeddings
    spk_id = "reference_speaker"
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    print("Speaker embeddings prepared")

    # Run parameter sweep
    all_metrics = []

    for idx, (temp, tp, tk) in enumerate(combos, 1):
        tag = f"t{temp}_p{tp}_k{tk}"
        output_file = os.path.join(out_dir, f'{tag}.wav')

        print(f"\n{'─' * 70}")
        print(f"[{idx}/{len(combos)}] temperature={temp}  top_p={tp}  top_k={tk}")
        print(f"{'─' * 70}")

        try:
            metrics = synthesize_offline(
                cosyvoice=cosyvoice,
                text=SYNTHESIS_TEXT,
                prompt_text=prompt_text,
                spk_id=spk_id,
                sample_rate=sample_rate,
                output_path=output_file,
                temperature=temp,
                top_p=tp,
                top_k=tk,
            )

            metrics['temperature'] = temp
            metrics['top_p'] = tp
            metrics['top_k'] = tk
            all_metrics.append(metrics)

            print(f"  Saved: {output_file}")
            print(f"  Duration: {metrics['audio_duration']:.2f}s  "
                  f"Time: {metrics['total_time']:.2f}s  "
                  f"RTF: {metrics['rtf']:.3f}")

        except Exception as e:
            logger.error(f"Error with {tag}: {e}", exc_info=True)
            continue

        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Summary table
    if all_metrics:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'temp':>6} {'top_p':>6} {'top_k':>6} | {'duration':>9} {'time':>8} {'rtf':>7}")
        print("-" * 55)
        for m in all_metrics:
            print(
                f"{m['temperature']:>6.2f} {m['top_p']:>6.2f} {m['top_k']:>6d} | "
                f"{m['audio_duration']:>8.2f}s {m['total_time']:>7.2f}s {m['rtf']:>7.3f}"
            )

    print(f"\nResults saved to: {out_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
