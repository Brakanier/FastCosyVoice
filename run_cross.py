#!/usr/bin/env python3
"""
FastCosyVoice3 TTS - Cross-lingual inference test

Tests both cross-lingual methods:
1. inference_cross_lingual_stream() - streaming with parallel pipeline
2. inference_cross_lingual() - offline (non-streaming)

Cross-lingual mode synthesizes text in any language using the voice from
a reference audio, without requiring a text transcription of that audio.
The LLM receives only the target text (no prompt text or speech tokens),
while Flow uses the speaker embedding and speech features for voice cloning.
"""

import sys
import time
import os
import logging
import wave
from pathlib import Path

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

# Reference audio (voice donor) -- no transcription needed for cross-lingual
REFERENCE_AUDIO = 'refs/audio.wav'

OUTPUT_DIR = 'output/run_cross'

USE_TRT_FLOW = True
USE_TRT_LLM = True
TRT_LLM_DTYPE = 'bfloat16'
TRT_LLM_KV_CACHE_TOKENS = 8192

LLM_MODEL_NAME = 'llm.rl.pt'

TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 25

# Text for streaming cross-lingual synthesis
STREAM_TEXT = (
    "And then later on, fully acquiring that company. "
    "So keeping management in line, interest in line with the asset "
    "that's coming into the family is a reason why sometimes we don't buy the whole thing."
)

# Text for offline cross-lingual synthesis
OFFLINE_TEXT = (
    "The most important thing in communication is hearing what isn't said. "
    "The art of reading between the lines is a lifelong quest of the wise."
)


def save_pcm_as_wav(pcm_chunks: list[bytes], sample_rate: int, output_path: str) -> float:
    """Save raw PCM int16 chunks as a WAV file. Returns audio duration in seconds."""
    full_pcm = b''.join(pcm_chunks)
    if not full_pcm:
        return 0.0
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(full_pcm)
    return len(full_pcm) / 2 / sample_rate


def test_streaming(cosyvoice: FastCosyVoice3, sample_rate: int):
    """Test cross-lingual streaming synthesis."""
    print("\n" + "=" * 70)
    print("[1/2] Cross-lingual STREAMING")
    print("=" * 70)
    print(f"Text: {STREAM_TEXT}")

    output_path = os.path.join(OUTPUT_DIR, 'cross_lingual_stream.wav')
    audio_chunks: list[bytes] = []
    chunk_count = 0
    first_chunk_time = None

    start_time = time.time()

    with torch.inference_mode():
        for pcm_bytes in cosyvoice.inference_cross_lingual_stream(
            tts_text=STREAM_TEXT,
            prompt_wav=REFERENCE_AUDIO,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
        ):
            chunk_count += 1
            if first_chunk_time is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                first_chunk_time = time.time() - start_time
            audio_chunks.append(pcm_bytes)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    audio_duration = save_pcm_as_wav(audio_chunks, sample_rate, output_path)
    rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

    print(f"\nSaved: {output_path}")
    print(f"TTFB:       {first_chunk_time or 0:.3f} sec")
    print(f"Total time: {total_time:.3f} sec")
    print(f"Duration:   {audio_duration:.3f} sec")
    print(f"RTF:        {rtf:.3f}")
    print(f"Chunks:     {chunk_count}")

    return {'ttfb': first_chunk_time or 0, 'total_time': total_time,
            'audio_duration': audio_duration, 'rtf': rtf}


def test_offline(cosyvoice: FastCosyVoice3, sample_rate: int):
    """Test cross-lingual offline (non-streaming) synthesis."""
    print("\n" + "=" * 70)
    print("[2/2] Cross-lingual OFFLINE")
    print("=" * 70)
    print(f"Text: {OFFLINE_TEXT}")

    output_path = os.path.join(OUTPUT_DIR, 'cross_lingual_offline.wav')
    audio_chunks: list[bytes] = []
    segment_count = 0

    start_time = time.time()

    with torch.inference_mode():
        for pcm_bytes in cosyvoice.inference_cross_lingual(
            tts_text=OFFLINE_TEXT,
            prompt_wav=REFERENCE_AUDIO,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
        ):
            segment_count += 1
            audio_chunks.append(pcm_bytes)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.time() - start_time
    audio_duration = save_pcm_as_wav(audio_chunks, sample_rate, output_path)
    rtf = total_time / audio_duration if audio_duration > 0 else float('inf')

    print(f"\nSaved: {output_path}")
    print(f"Total time: {total_time:.3f} sec")
    print(f"Duration:   {audio_duration:.3f} sec")
    print(f"RTF:        {rtf:.3f}")
    print(f"Segments:   {segment_count}")

    return {'total_time': total_time, 'audio_duration': audio_duration, 'rtf': rtf}


def main():
    print("=" * 70)
    print("FastCosyVoice3 - Cross-Lingual Synthesis Test")
    print("=" * 70)

    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}", exc_info=True)
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"\nReference audio: {REFERENCE_AUDIO}")
    print(f"Loading FastCosyVoice3...")

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
    print(f"Model loaded in {time.time() - load_start:.2f} sec")

    if not (USE_TRT_LLM and cosyvoice.trt_llm_loaded):
        qwen2_model = cosyvoice.model.llm.llm.model.model
        cosyvoice.model.llm.llm.model.model = torch.compile(qwen2_model, mode="default")
        logger.info("torch.compile applied to LLM")

    sample_rate = cosyvoice.sample_rate

    # --- Quick warmup (one short cross-lingual generation) ---
    print("\nWarming up...")
    with torch.inference_mode():
        for _ in cosyvoice.inference_cross_lingual_stream(
            tts_text="Warmup text.",
            prompt_wav=REFERENCE_AUDIO,
        ):
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup done")

    # --- Run tests ---
    try:
        m_stream = test_streaming(cosyvoice, sample_rate)
    except Exception as e:
        logger.error(f"Streaming test failed: {e}", exc_info=True)
        m_stream = None

    try:
        m_offline = test_offline(cosyvoice, sample_rate)
    except Exception as e:
        logger.error(f"Offline test failed: {e}", exc_info=True)
        m_offline = None

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    llm_backend = "TensorRT-LLM" if (USE_TRT_LLM and cosyvoice.trt_llm_loaded) else "PyTorch"
    flow_backend = "TensorRT" if USE_TRT_FLOW else "PyTorch"
    print(f"LLM: {llm_backend}, Flow: {flow_backend}")
    print()

    if m_stream:
        print(f"Streaming:  RTF={m_stream['rtf']:.3f}  TTFB={m_stream['ttfb']:.3f}s  "
              f"audio={m_stream['audio_duration']:.1f}s  time={m_stream['total_time']:.1f}s")
    if m_offline:
        print(f"Offline:    RTF={m_offline['rtf']:.3f}  "
              f"audio={m_offline['audio_duration']:.1f}s  time={m_offline['total_time']:.1f}s")

    print(f"\nResults: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
