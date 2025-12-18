#!/usr/bin/env python3
"""
CosyVoice3 TTS - Упрощённый скрипт для streaming инференса с замером метрик

Использует метод inference_zero_shot для генерации с клонированием голоса.
Использует TRT и FP16 для оптимизации.

Метрики:
- TTFB (Time To First Byte): время до получения первого чанка аудио
- RTF (Real-Time Factor): время_синтеза / длительность_аудио (< 1.0 = быстрее реалтайма)
- Длительность итогового аудио
- Общее время генерации
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

# Оптимизация для matmul операций
torch.set_float32_matmul_precision('high')

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Директория с моделью
MODEL_DIR = 'pretrained_models/Fun-CosyVoice3-0.5B'

# Референсный аудио файл (3-10 сек, чистая запись)
REFERENCE_AUDIO = 'refs/yaga.wav'

# Директория для результатов
OUTPUT_DIR = 'output'

# Инструкция для модели
INSTRUCTION = "You are a helpful assistant."

# Тексты для синтеза
SYNTHESIS_TEXTS = [
    "Привет! Это тестовый синтез русского текста с использованием модели CosyVoice3.",
    "Второй пример текста для генерации. [cough] [cough] Блять! Надо бы бросать курить",
    "И третий текст [laughter] для демонстрации [laughter] возможности генерировать [laughter] смехуёчки.",
]


def load_prompt_text(audio_path: str, instruction: str = INSTRUCTION) -> str:
    """
    Загружает транскрипцию из txt файла и формирует prompt_text.
    
    Формат prompt_text: "{instruction}<|endofprompt|>{транскрипция}"
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
    Выполняет streaming синтез текста через zero_shot и возвращает метрики.
    
    Args:
        prompt_text: Транскрипция референсного аудио в формате "{instruction}<|endofprompt|>{транскрипция}"
    
    Returns:
        dict с ключами: ttfb, total_time, audio_duration, rtf, chunk_count
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
    
    # Объединяем чанки и сохраняем
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
    
    # Проверяем наличие модели
    if not os.path.exists(MODEL_DIR):
        logger.error(f"Модель не найдена: {MODEL_DIR}", exc_info=True)
        return
    
    # Проверяем наличие референсного аудио
    if not os.path.exists(REFERENCE_AUDIO):
        logger.error(f"Референсный аудио не найден: {REFERENCE_AUDIO}", exc_info=True)
        return
    
    # Создаём директорию для результатов
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Загружаем prompt_text из txt файла рядом с аудио
    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)
    
    print(f"\n🎤 Референсный аудио: {REFERENCE_AUDIO}")
    print(f"📝 Текстов для синтеза: {len(SYNTHESIS_TEXTS)}")
    
    # Загрузка модели (TRT и FP16)
    print("\n🔧 Загрузка модели...")
    load_start = time.time()
    
    cosyvoice = CosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_vllm=False,
        load_trt=True,
    )
    
    load_time = time.time() - load_start
    print(f"✅ Модель загружена за {load_time:.2f} сек")
    
    # Диагностика dtype
    llm_dtype = next(cosyvoice.model.llm.parameters()).dtype
    flow_dtype = next(cosyvoice.model.flow.parameters()).dtype
    hift_dtype = next(cosyvoice.model.hift.parameters()).dtype
    print(f"📊 LLM dtype: {llm_dtype}, Flow dtype: {flow_dtype}, HiFT dtype: {hift_dtype}")
    
    sample_rate = cosyvoice.sample_rate
    print(f"📊 Sample rate: {sample_rate} Hz")
    
    # Подготовка эмбеддингов спикера (один раз)
    print("\n🎯 Подготовка эмбеддингов спикера...")
    spk_id = "reference_speaker"
    embed_start = time.time()
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    embed_time = time.time() - embed_start
    print(f"✅ Эмбеддинги подготовлены за {embed_time:.3f} сек")
    
    # Сводка по всем текстам
    all_metrics = []
    
    # Генерация всех текстов
    for idx, text in enumerate(SYNTHESIS_TEXTS, 1):
        print("\n" + "=" * 70)
        print(f"📄 Текст {idx}/{len(SYNTHESIS_TEXTS)}")
        print("=" * 70)
        print(f"📝 {text[:80]}{'...' if len(text) > 80 else ''}")
        
        output_file = os.path.join(OUTPUT_DIR, f'output_{idx:02d}.wav')
        
        try:
            metrics = synthesize_streaming(
                cosyvoice=cosyvoice,
                text=text,
                prompt_text=prompt_text,  # транскрипция референсного аудио
                spk_id=spk_id,
                sample_rate=sample_rate,
                output_path=output_file,
            )
            
            all_metrics.append(metrics)
            
            print(f"\n💾 Сохранено: {output_file}")
            print("\n📊 МЕТРИКИ:")
            print("-" * 40)
            print(f"⚡ TTFB:             {metrics['ttfb']:.3f} сек")
            print(f"⏱️  Общее время:      {metrics['total_time']:.3f} сек")
            print(f"🎵 Длительность:     {metrics['audio_duration']:.3f} сек")
            print(f"📈 RTF:              {metrics['rtf']:.3f}")
            print(f"📦 Чанков:           {metrics['chunk_count']}")
            
            if metrics['rtf'] < 1.0:
                print(f"✅ Быстрее реалтайма в {1/metrics['rtf']:.1f}x")
            else:
                print(f"⚠️  Медленнее реалтайма в {metrics['rtf']:.1f}x")
                
        except Exception as e:
            logger.error(f"Ошибка при синтезе текста #{idx}: {e}", exc_info=True)
            continue
    
    # Итоговая сводка
    if all_metrics:
        print("\n" + "=" * 70)
        print("📊 ИТОГОВАЯ СВОДКА")
        print("=" * 70)
        
        avg_ttfb = sum(m['ttfb'] for m in all_metrics) / len(all_metrics)
        avg_rtf = sum(m['rtf'] for m in all_metrics) / len(all_metrics)
        total_audio = sum(m['audio_duration'] for m in all_metrics)
        total_time = sum(m['total_time'] for m in all_metrics)
        
        print(f"Средний TTFB:        {avg_ttfb:.3f} сек")
        print(f"Средний RTF:         {avg_rtf:.3f}")
        print(f"Общая длительность:  {total_audio:.3f} сек")
        print(f"Общее время:         {total_time:.3f} сек")
    
    print("\n" + "=" * 70)
    print("✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"\n📁 Результаты: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()

