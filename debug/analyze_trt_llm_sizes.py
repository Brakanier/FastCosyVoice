#!/usr/bin/env python3
"""
Analyze real input/output sizes for TRT-LLM optimization.

This script measures actual token lengths to determine optimal
max_input_len, max_output_len, and max_num_tokens for TRT-LLM.
"""

import sys
sys.path.append('third_party/Matcha-TTS')

import os
import torch
from hyperpyyaml import load_hyperpyyaml

# Configuration
MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'
REFERENCE_AUDIO = 'refs/audio.wav'
INSTRUCTION = "You are a helpful assistant."

# Test texts of various lengths
TEST_TEXTS = [
    # Short
    "Привет! Как дела?",
    # Medium
    "Это тестовый синтез русского текста с использованием модели CosyVoice3.",
    # Long - single chunk after split_paragraph (token_max_n=80)
    """Начало есть время, когда следует позаботиться о том, чтобы все было отмерено и уравновешено. Начало есть время, когда следует позаботиться о том, чтобы все было отмерено и уравновешено. Начало есть время, когда следует позаботиться о том, чтобы все было отмерено и уравновешено.""",
    # Very long - but split_paragraph will split this into ~80 token chunks!
    # So we need to test individual chunks, not full text
]


def load_prompt_text(audio_path: str, instruction: str = INSTRUCTION) -> str:
    """Load transcription from txt file."""
    txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as f:
        transcription = f.read().strip()
    return f"{instruction}<|endofprompt|>{transcription}"


def main():
    print("=" * 70)
    print("TRT-LLM Size Analysis for CosyVoice3")
    print("=" * 70)
    
    # Load config
    hyper_yaml_path = os.path.join(MODEL_DIR, 'cosyvoice3.yaml')
    hf_llm_dir = os.path.join(MODEL_DIR, 'CosyVoice-BlankEN')
    
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': hf_llm_dir})
    
    # Get tokenizer (get_tokenizer is a factory function, call it to get actual tokenizer)
    tokenizer = configs['get_tokenizer']()
    
    # Config values
    speech_token_size = 6561  # from yaml
    token_frame_rate = 25     # 25 tokens/sec
    
    print(f"\n📋 Config from cosyvoice3.yaml:")
    print(f"   speech_token_size: {speech_token_size}")
    print(f"   token_frame_rate: {token_frame_rate} tokens/sec")
    
    # Load prompt text
    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)
    prompt_text_tokens = tokenizer.encode(prompt_text, allowed_special='all')
    
    print(f"\n📝 Prompt text analysis:")
    print(f"   Characters: {len(prompt_text)}")
    print(f"   Text tokens: {len(prompt_text_tokens)}")
    
    # Estimate prompt speech tokens (from reference audio duration)
    import torchaudio
    waveform, sr = torchaudio.load(REFERENCE_AUDIO)
    duration_sec = waveform.shape[1] / sr
    prompt_speech_tokens_est = int(duration_sec * token_frame_rate)
    
    print(f"\n🎤 Reference audio analysis:")
    print(f"   Duration: {duration_sec:.2f} sec")
    print(f"   Estimated speech tokens: {prompt_speech_tokens_est}")
    
    print("\n" + "=" * 70)
    print("📊 Input/Output Size Analysis per Text Chunk")
    print("=" * 70)
    
    max_input_len_seen = 0
    max_output_len_seen = 0
    
    for i, text in enumerate(TEST_TEXTS):
        # Tokenize text
        text_tokens = tokenizer.encode(text, allowed_special='all')
        
        # Input structure: [sos, spk_emb, prompt_text, tts_text, task_id, prompt_speech_tokens]
        # In TRT-LLM merged model, all are token IDs
        input_len = (
            1 +                          # sos
            len(prompt_text_tokens) +    # prompt_text (includes instruction)
            len(text_tokens) +           # tts_text
            1 +                          # task_id
            prompt_speech_tokens_est     # prompt_speech_tokens
        )
        
        # Output: speech tokens
        # Estimate based on text length and typical speech rate
        # ~25 speech tokens per second, ~5-10 chars per second in Russian
        chars_per_sec = 7  # conservative estimate
        estimated_duration = len(text) / chars_per_sec
        output_len = int(estimated_duration * token_frame_rate * 1.5)  # 1.5x margin
        
        # Also calculate based on token ratios from config
        min_output = len(text_tokens) * 2   # min_token_text_ratio
        max_output = len(text_tokens) * 20  # max_token_text_ratio
        
        max_input_len_seen = max(max_input_len_seen, input_len)
        max_output_len_seen = max(max_output_len_seen, output_len)
        
        print(f"\n{'─' * 50}")
        print(f"Text {i+1}: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"   Characters: {len(text)}")
        print(f"   Text tokens: {len(text_tokens)}")
        print(f"   Estimated input_len: {input_len}")
        print(f"   Estimated output_len: {output_len} (range: {min_output}-{max_output})")
    
    # Add safety margin
    recommended_input = ((max_input_len_seen + 127) // 128) * 128  # Round up to 128
    recommended_output = ((max_output_len_seen + 127) // 128) * 128
    
    # Ensure minimum reasonable values
    recommended_input = max(recommended_input, 512)
    recommended_output = max(recommended_output, 512)
    
    # IMPORTANT: Consider worst case from config!
    # max_token_text_ratio = 20 means up to 20 speech tokens per text token
    # With token_max_n = 80 text tokens, worst case = 80 * 20 = 1600 speech tokens
    worst_case_output = 80 * 20  # 1600
    
    # Use the larger of estimated and worst case (with some margin)
    recommended_output_safe = max(recommended_output, ((worst_case_output + 255) // 256) * 256)  # 1792
    
    print("\n" + "=" * 70)
    print("📊 RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\n🔍 Observed maximums:")
    print(f"   max_input_len seen:  {max_input_len_seen}")
    print(f"   max_output_len seen: {max_output_len_seen}")
    
    print(f"\n⚠️  Worst case from config (max_token_text_ratio=20, token_max_n=80):")
    print(f"   worst_case_output: {worst_case_output} (80 * 20)")
    
    print(f"\n✅ Recommended TRT-LLM parameters:")
    print(f"   max_input_len:  {recommended_input} (based on observed + margin)")
    print(f"   max_output_len: {recommended_output_safe} (based on worst case)")
    print(f"   max_num_tokens: {recommended_input + recommended_output_safe}")
    
    print(f"\n📊 Current vs Recommended:")
    print(f"   {'Parameter':<20} {'Current':<10} {'Recommended':<12} {'Reduction':<10}")
    print(f"   {'-' * 52}")
    print(f"   {'max_input_len':<20} {4096:<10} {recommended_input:<12} {4096/recommended_input:.1f}x")
    print(f"   {'max_output_len':<20} {2048:<10} {recommended_output_safe:<12} {2048/recommended_output_safe:.1f}x")
    print(f"   {'max_num_tokens':<20} {16384:<10} {recommended_input + recommended_output_safe:<12} {16384/(recommended_input + recommended_output_safe):.1f}x")
    
    print(f"\n💾 Estimated VRAM savings:")
    print(f"   KV-cache reduction: ~{(4096 + 2048) / (recommended_input + recommended_output_safe):.1f}x smaller")
    print(f"   This could save ~2-3 GB of VRAM!")
    
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT: To apply these changes:")
    print("=" * 70)
    print("""
1. Update fastcosyvoice/cosyvoice.py:
   - Change runner_kwargs max_input_len and max_output_len
   - Change trtllm-build --max_input_len and --max_num_tokens

2. Delete existing TRT-LLM engines to force rebuild:
   rm -rf pretrained_models/Fun-CosyVoice3-0.5B/trt_llm_*

3. Re-run your script - engines will be rebuilt with new sizes
""")


if __name__ == '__main__':
    main()

