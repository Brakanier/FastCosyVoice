#!/usr/bin/env python3
"""
CosyVoice3 TTS - Simplified script for streaming inference with metrics measurement

Uses inference_zero_shot method for generation with voice cloning.
Uses TRT and FP16 for optimization.

Metrics:
- TTFB (Time To First Byte): time until first audio chunk is received
- RTF (Real-Time Factor): synthesis_time / audio_duration (< 1.0 = faster than real-time)
- Final audio duration
- Total generation time
"""

import sys
import time
import os
import logging
from pathlib import Path

sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice3

# Optimization for matmul operations
torch.set_float32_matmul_precision('high')

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model directory
MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'

# Reference audio file (3-10 sec, clean recording)
REFERENCE_AUDIO = 'refs/audio.wav'

# Output directory
OUTPUT_DIR = 'output/run_basic'

# Instruction for the model
INSTRUCTION = "You are a helpful assistant."

# Texts for synthesis
SYNTHESIS_TEXTS = [
    "Привет! Это тестовый синтез русского текста с использованием модели CosyVoice3.",
    "Второй пример текста для генерации. [cough] [cough] Блять! Надо бы бросать курить",
    "И третий текст [laughter] для демонстрации [laughter] возможности генерировать [laughter] [laughter] смехуёчки.",
]


def load_prompt_text(audio_path: str, instruction: str = INSTRUCTION) -> str:
    """
    Loads transcription from txt file and forms prompt_text.
    
    Format prompt_text: "{instruction}<|endofprompt|>{transcription}"
    """
    txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        transcription = f.read().strip()
    
    return f"{instruction}<|endofprompt|>{transcription}"


def synthesize_streaming(
    cosyvoice: CosyVoice3,
    text: str,
    prompt_text: str,
    spk_id: str,
    sample_rate: int,
    output_path: str
) -> dict:
    """
    Performs streaming synthesis of text via zero_shot and returns metrics.
    
    Args:
        prompt_text: Transcription of reference audio in format "{instruction}<|endofprompt|>{transcription}"
    
    Returns:
        dict with keys: ttfb, total_time, audio_duration, rtf, chunk_count
    """
    start_time = time.time()
    first_chunk_time = None
    audio_chunks = []
    chunk_count = 0
    
    for model_output in cosyvoice.inference_zero_shot(
        tts_text=text,
        prompt_text=prompt_text,
        prompt_wav=REFERENCE_AUDIO,
        zero_shot_spk_id=spk_id,
        stream=True,
    ):
        chunk_count += 1
        
        if first_chunk_time is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            first_chunk_time = time.time() - start_time
        
        speech = model_output['tts_speech']
        audio_chunks.append(speech)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    # Concatenate chunks and save
    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=1)
        torchaudio.save(output_path, full_audio, sample_rate)
        audio_duration = full_audio.shape[1] / sample_rate
    else:
        audio_duration = 0.0
    
    rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
    
    return {
        'ttfb': first_chunk_time or 0.0,
        'total_time': total_time,
        'audio_duration': audio_duration,
        'rtf': rtf,
        'chunk_count': chunk_count,
    }


def main():
    print("=" * 70)
    print("CosyVoice3 TTS - Streaming Inference (zero_shot)")
    print("=" * 70)
        
    # Check if reference audio exists
    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Reference audio not found: {REFERENCE_AUDIO}", exc_info=True)
        return
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load prompt_text from txt file next to audio
    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)
    
    print(f"\n🎤 Reference audio: {REFERENCE_AUDIO}")
    print(f"📝 Texts for synthesis: {len(SYNTHESIS_TEXTS)}")
    
    # Load model (TRT and FP16)
    print("\n🔧 Loading model...")
    load_start = time.time()
    
    cosyvoice = CosyVoice3(
        model_dir=MODEL_DIR,
        fp16=False,
        load_vllm=False,
        load_trt=False,
    )
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f} sec")
    
    # dtype diagnostics
    llm_dtype = next(cosyvoice.model.llm.parameters()).dtype
    flow_dtype = next(cosyvoice.model.flow.parameters()).dtype
    hift_dtype = next(cosyvoice.model.hift.parameters()).dtype
    print(f"📊 LLM dtype: {llm_dtype}, Flow dtype: {flow_dtype}, HiFT dtype: {hift_dtype}")
    
    sample_rate = cosyvoice.sample_rate
    print(f"📊 Sample rate: {sample_rate} Hz")
    
    # Prepare speaker embeddings (once)
    print("\n🎯 Preparing speaker embeddings...")
    spk_id = "reference_speaker"
    embed_start = time.time()
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    embed_time = time.time() - embed_start
    print(f"✅ Embeddings prepared in {embed_time:.3f} sec")
    
    # Summary for all texts
    all_metrics = []
    
    # Generate all texts
    for idx, text in enumerate(SYNTHESIS_TEXTS, 1):
        print("\n" + "=" * 70)
        print(f"📄 Text {idx}/{len(SYNTHESIS_TEXTS)}")
        print("=" * 70)
        print(f"📝 {text[:80]}{'...' if len(text) > 80 else ''}")
        
        output_file = os.path.join(OUTPUT_DIR, f'output_{idx:02d}.wav')
        
        try:
            metrics = synthesize_streaming(
                cosyvoice=cosyvoice,
                text=text,
                prompt_text=prompt_text,  # transcription of reference audio
                spk_id=spk_id,
                sample_rate=sample_rate,
                output_path=output_file,
            )
            
            all_metrics.append(metrics)
            
            print(f"\n💾 Saved: {output_file}")
            print("\n📊 METRICS:")
            print("-" * 40)
            print(f"⚡ TTFB:             {metrics['ttfb']:.3f} sec")
            print(f"⏱️  Total time:       {metrics['total_time']:.3f} sec")
            print(f"🎵 Duration:         {metrics['audio_duration']:.3f} sec")
            print(f"📈 RTF:              {metrics['rtf']:.3f}")
            print(f"📦 Chunks:           {metrics['chunk_count']}")
            
            if metrics['rtf'] < 1.0:
                print(f"✅ Faster than real-time by {1/metrics['rtf']:.1f}x")
            else:
                print(f"⚠️  Slower than real-time by {metrics['rtf']:.1f}x")
                
        except Exception as e:
            logger.error(f"Error synthesizing text #{idx}: {e}", exc_info=True)
            continue
    
    # Final summary
    if all_metrics:
        print("\n" + "=" * 70)
        print("📊 FINAL SUMMARY")
        print("=" * 70)
        
        avg_ttfb = sum(m['ttfb'] for m in all_metrics) / len(all_metrics)
        avg_rtf = sum(m['rtf'] for m in all_metrics) / len(all_metrics)
        total_audio = sum(m['audio_duration'] for m in all_metrics)
        total_time = sum(m['total_time'] for m in all_metrics)
        
        print(f"Average TTFB:        {avg_ttfb:.3f} sec")
        print(f"Average RTF:         {avg_rtf:.3f}")
        print(f"Total duration:      {total_audio:.3f} sec")
        print(f"Total time:          {total_time:.3f} sec")
    
    print("\n" + "=" * 70)
    print("✅ GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

