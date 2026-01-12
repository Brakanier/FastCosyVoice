#!/usr/bin/env python3
"""
CosyVoice3 TTS - Simplified script for streaming inference with metrics measurement

Uses inference_zero_shot method for generation with voice cloning.
Applies torch.compile to accelerate LLM inference (~2x speedup).

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


def get_gpu_memory_stats() -> dict:
    """
    Returns GPU memory usage statistics.
    
    Returns:
        dict with keys: allocated_gb, reserved_gb, max_allocated_gb
    """
    import torch
    if not torch.cuda.is_available():
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'max_allocated_gb': 0.0}
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
    }


def print_gpu_memory(label: str) -> None:
    """Prints current GPU memory state with a label."""
    stats = get_gpu_memory_stats()
    print(f"\n📊 GPU Memory [{label}]:")
    print(f"   Allocated: {stats['allocated_gb']:.2f} GB")
    print(f"   Reserved: {stats['reserved_gb']:.2f} GB")
    print(f"   Peak allocated: {stats['max_allocated_gb']:.2f} GB")

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice3

# Optimization for torch.compile
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
OUTPUT_DIR = 'output/run'

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


def apply_torch_compile(cosyvoice: CosyVoice3) -> None:
    """
    Applies torch.compile to LLM model to accelerate inference.
    
    Compiles internal Qwen2ForCausalLM.model (Qwen2Model),
    which is used in forward_one_step for auto-generation.
    """
    # Path to Qwen2Model: cosyvoice.model.llm.llm.model.model
    # llm - CosyVoice3LM
    # llm.llm - Qwen2Encoder  
    # llm.llm.model - Qwen2ForCausalLM
    # llm.llm.model.model - Qwen2Model (what is actually called in forward_one_step)
    
    qwen2_model = cosyvoice.model.llm.llm.model.model
    logger.info(f"Compiling Qwen2Model: {type(qwen2_model).__name__}")
    
    compiled_model = torch.compile(qwen2_model, mode="default")
    cosyvoice.model.llm.llm.model.model = compiled_model
    
    logger.info("torch.compile applied to LLM")


def warmup_model(
    cosyvoice: CosyVoice3,
    prompt_text: str,
    spk_id: str,
) -> None:
    """
    Warms up the model by generating tokens to compile all execution paths.
    
    torch.compile creates different kernels for different input sizes,
    so the model needs to be warmed up on texts of different lengths.
    
    Args:
        cosyvoice: Initialized CosyVoice3 model
        prompt_text: Prompt text for generation
        spk_id: Speaker ID (should already be added via add_zero_shot_spk)
    """
    # Texts of different lengths to cover different input sizes
    warmup_texts = [
        # Short text (~50-100 LLM tokens)
        "Hello! How are you?",
        # Medium text (~100-200 LLM tokens)  
        "This is a test synthesis of medium-length text for model warmup.",
        # Long text (~200-400 LLM tokens)
        "This is a longer text for warmup. " * 3,
        # Very long text (~400+ LLM tokens)
        "Warming up the model on a long text for compilation. " * 5,
    ]
    
    warmup_start = time.time()
    
    # First pass - main compilation
    logger.info("Warmup: first pass (kernel compilation)...")
    for i, text in enumerate(warmup_texts):
        logger.info(f"  Warmup text {i+1}/{len(warmup_texts)}: {len(text)} characters")
        for _ in cosyvoice.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
            stream=True,
        ):
            pass  # Just generate all chunks
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Second pass - ensure all paths are compiled
    logger.info("Warmup: second pass (stabilization)...")
    for text in warmup_texts:
        for _ in cosyvoice.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
            stream=True,
        ):
            pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_time = time.time() - warmup_start
    logger.info(f"Warmup completed in {warmup_time:.2f} sec")


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
        prompt_text: Reference audio transcription in format "{instruction}<|endofprompt|>{transcription}"
    
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
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load prompt_text from txt file next to audio
    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)
    
    print(f"\n🎤 Reference audio: {REFERENCE_AUDIO}")
    print(f"📝 Texts for synthesis: {len(SYNTHESIS_TEXTS)}")
    
    # Load model (without optimizations)
    print("\n🔧 Loading model...")
    load_start = time.time()
    
    cosyvoice = CosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_vllm=False,
        load_trt=True,
    )
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f} sec")
    
    print_gpu_memory("after model loading")
    
    # dtype diagnostics
    llm_dtype = next(cosyvoice.model.llm.parameters()).dtype
    flow_dtype = next(cosyvoice.model.flow.parameters()).dtype
    hift_dtype = next(cosyvoice.model.hift.parameters()).dtype
    print(f"📊 LLM dtype: {llm_dtype}, Flow dtype: {flow_dtype}, HiFT dtype: {hift_dtype}")
    
    sample_rate = cosyvoice.sample_rate
    print(f"📊 Sample rate: {sample_rate} Hz")
    
    # Apply torch.compile to LLM
    print("\n⚡ Applying torch.compile to LLM...")
    compile_start = time.time()
    apply_torch_compile(cosyvoice)
    compile_time = time.time() - compile_start
    print(f"✅ torch.compile applied in {compile_time:.3f} sec")
    
    # Prepare speaker embeddings (once)
    print("\n🎯 Preparing speaker embeddings...")
    spk_id = "reference_speaker"
    embed_start = time.time()
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    embed_time = time.time() - embed_start
    print(f"✅ Embeddings prepared in {embed_time:.3f} sec")
    
    # Model warmup (graph compilation)
    print("\n🔥 Warming up model (compiling graphs for different text lengths)...")
    warmup_model(cosyvoice, prompt_text, spk_id)
    print("✅ Model warmed up and ready")
    
    print_gpu_memory("after warmup")
    
    # Reset peak memory counter to measure only generation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
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
                prompt_text=prompt_text,  # reference audio transcription
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
    
    print_gpu_memory("after generation")
    
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
    print("✅ GENERATION COMPLETED!")
    print("=" * 70)
    print(f"\n📁 Results: {OUTPUT_DIR}/")
    
    print_gpu_memory("at the end")


if __name__ == '__main__':
    main()

