#!/usr/bin/env python3
"""
FastCosyVoice3 TTS - Non-streaming (offline) inference with metrics measurement

Uses FastCosyVoice3 with TensorRT acceleration:
- LLM: TensorRT-LLM (~3x speedup) or PyTorch with torch.compile
- Flow: TensorRT (~2.5x speedup)
- Hift: PyTorch (f0_predictor on CPU)

Non-streaming mode generates all speech tokens first, then converts to audio.
This has higher latency but can be simpler for batch processing.

Metrics:
- RTF (Real-Time Factor): synthesis_time / audio_duration (< 1.0 = faster than real-time)
- Final audio duration
- Total generation time
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


def get_gpu_memory_stats() -> dict:
    """
    Returns GPU memory usage statistics.
    
    Returns:
        dict with keys: allocated_gb, reserved_gb, max_allocated_gb
    """
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
    print(f"   Peak Allocated: {stats['max_allocated_gb']:.2f} GB")
# Optimization for torch.compile (if used)
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
OUTPUT_DIR = 'output/run_offline'

# Instruction for the model
INSTRUCTION = "You are a helpful assistant."

# TensorRT settings
USE_TRT_FLOW = True       # TensorRT for Flow decoder (~2.5x speedup)
USE_TRT_LLM = True        # TensorRT-LLM for LLM (~3x speedup)
TRT_LLM_DTYPE = 'bfloat16'  # bfloat16/float16/float32
# Fraction of VRAM that TensorRT-LLM pre-reserves for KV-cache.
# This is often the main memory consumer. Decrease if VRAM usage is high (e.g., 0.10-0.20).
TRT_LLM_KV_CACHE_FRACTION = 0.10

# Inference wrapper without autograd (reduces allocations and graph leak risk)
USE_INFERENCE_MODE = True

# Text for synthesis
SYNTHESIS_TEXT = """
Начало есть время, когда следует позаботиться о том, чтобы все было отмерено и уравновешено. Это знает каждая сестра Бене Гессерит. Итак, приступая к изучению жизни Муад'Диба, прежде всего правильно представьте время его: рожден в пятьдесят седьмой год правления Падишах-Императора Шаддама IV. И с особым вниманием отнеситесь к его месту в пространстве: планете Арракис. Пусть не смутит вас то, что родился он на Каладане и первые пятнадцать лет своей жизни провел на этой планете: Арракис, часто называемой также Дюной, – вот место Муад'Диба, вовеки.

Из учебника «Жизнь Муад'Диба» принцессы Ирулан

За неделю до отлета на Арракис, когда суета приготовлений и сборов достигла апогея, превратившись в настоящее безумие, какая-то сморщенная старуха пришла к матери Пауля.

Над замком Каладан стояла теплая ночь, но из древних каменных стен, двадцать шесть поколений служивших роду Атрейдесов, как всегда перед сменой погоды, выступил тонкий, прохладный налет влаги.

Старуху впустили через боковую дверь, провели сводчатым коридором мимо комнаты Пауля, и она, заглянув в нее, увидела лежащего в постели юного наследника.

В тусклом свете плавающей лампы, притушенной и висящей в силовом поле у самого пола, проснувшийся мальчик увидел в дверях грузную женщину – та стояла на шаг впереди его матери. Старуха походила на ведьму: свалявшаяся паутина волос, подобно капюшону, затеняла лицо, на котором ярко сверкали глаза.

– Не маловат ли он для своих лет, Джессика? – спросила старуха. У нее была одышка, а резкий, дребезжащий голос звучал как расстроенный балисет.

Мать Пауля ответила своим мягким контральто:

– Все Атрейдесы взрослеют поздно, Преподобная.

– Слыхала, – проскрипела старуха. – Но ему уже пятнадцать.

– Да, Преподобная.

– Ага, он проснулся и слушает! – Старуха всмотрелась в лицо мальчика. – Притворяется, маленький хитрец! Ну да, для правителя хитрость не порок… А если он и впрямь Квисатц Хадерах – тогда… впрочем, посмотрим.

Пауль, укрывшись в тени своего ложа, смотрел на нее сквозь прикрытые веки. Ему казалось, что два сверкающих овала – глаза старухи – увеличились и засияли внутренним светом, встретившись с его взглядом.

– Спи, спи пока спокойно, притворщик, – проговорила старуха. – Выспись как следует: завтра тебе понадобятся все силы, какие у тебя есть… чтобы встретить мой гом джаббар…

С этим она и удалилась, вытеснив мать Пауля в коридор, и захлопнула дверь.

Пауль лежал и думал. Что такое гом джаббар?

Старуха была самым странным из всего, что он видел за эти дни перемен и суеты сборов.

Преподобная…

Эта «Преподобная» называла его мать просто «Джессика», словно простую служанку. А ведь его мать – Бене Гессерит, леди, наложница герцога Лето Атрейдеса, родившая ему наследника!

Но гом джаббар… что это? Нечто связанное с Арракисом? Что-то, что он должен узнать до того, как отправиться туда?

Он беззвучно повторил эти странные слова: «гом джаббар», «Квисатц Хадерах»…

Предстояло узнать столько нового! Арракис так отличался от Каладана, что голова Пауля шла кругом от обилия новых сведений.

Арракис – Дюна – Планета Пустыни.

Суфир Хават, старший мастер-асассин при дворе его отца, объяснял ему, что Харконнены, смертельные враги дома Атрейдес, восемьдесят лет властвовали над Арракисом – он был их квазиленным владением по контракту на добычу легендарного гериатрического снадобья, Пряности, меланжи – контракту, заключенному с Харконненами компанией КООАМ. Теперь Харконнены уходили, а на их место, но уже с полным леном, приходили Атрейдесы – и бесспорность победы герцога Лето Атрейдеса была очевидна. Хотя… Хават еще говорил, что в такой очевидности таится смертельная угроза, ибо герцог Лето слишком популярен в Ландсрааде Великих Правящих Домов. «А чужая слава – основа зависти владык», – сказал тогда Хават.

Арракис – Дюна – Планета Пустыни…

Пауль спал. Ему снилась какая-то пещера на Арракисе, молчаливые люди, скользящие в неясном свете плавающих в воздухе ламп. И тишина – торжественная тишина храма, нарушаемая только отчетливо отдающимися под сводами звуками часто падающих капель: кап-кап-кап… Пауль даже в забытьи чувствовал, что не забудет это видение – пробуждаясь, он всегда помнил сны, содержащие предсказание…

Видение становилось все более зыбким и наконец растаяло.

Пауль лежал в полудреме и думал. Замок Каладан, в котором он не знал игр со сверстниками, пожалуй, вовсе не заслуживал грусти при расставании. Доктор Юйэ, его учитель, намекнул, что на Арракисе классовые рамки кодекса Фафрелах не соблюдаются так строго, как здесь. Люди там живут в пустыне, где нет каидов и башаров Императора, чтобы командовать ими. Люди, подчиняющиеся лишь Воле Пустыни, фримены, «Свободные» – не внесенные в имперские переписи…

Арракис – Дюна – Планета Пустыни…

Пауль почувствовал охватившее его напряжение и применил один из приемов подчинения духа и тела, которым научила его мать. Три быстрых коротких вдоха – и привычная реакция: он словно поплыл, концентрируя при этом свое внутреннее «я»: …аорта расширяется… сознание сфокусировано… сознание контролируется полностью: я могу управлять сознанием, включать и выключать по собственному желанию… моя кровь насыщается кислородом и омывает им перегруженные участки… невозможно получить пищу, безопасность и свободу, пользуясь одним лишь инстинктом… разуму животного не дано выйти за пределы момента или осознать, что оно само может уничтожить свою добычу… животное разрушает, а не создает… удовольствия животного остаются на уровне чувственного восприятия, не возвышаясь до осознания… человек нуждается в системе координат для восприятия мира… концентрируя сознание, я создаю такую систему… единство тела следует за работой нервной и кровеносной систем – согласно нуждам самих клеток… все сущее, все предметы, все живое – все непостоянно… необходимо стремиться к постоянству изменчивости внутри себя…

Снова и снова повторялся этот урок в плывущем сознании Пауля.

Когда же сквозь шторы проник желтый свет утра, Пауль почувствовал его сквозь сомкнутые веки, открыл глаза и услышал, что в замке возобновилась суета. Увидел над собой знакомую резьбу потолочных балок…

Отворилась дверь, и в спальню заглянула мать: волосы цвета темной бронзы перевиты черной лентой, черты лица неподвижны и зеленые глаза торжественно-строги.

– Проснулся? – спросила, она. – Хорошо выспался?

– Да.

Пауль пристально смотрел на нее, пока мать выбирала одежду, примечая непривычную суровость, напряженные плечи… Никто другой не разглядел бы этого, но Джессика сама обучала его тайнам Бене Гессерит, заставляла обращать внимание на мельчайшие детали.
"""


def load_prompt_text(audio_path: str, instruction: str = INSTRUCTION) -> str:
    """
    Loads transcription from txt file and forms prompt_text.
    
    Format prompt_text: "{instruction}<|endofprompt|>{transcription}"
    """
    txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        transcription = f.read().strip()
    
    return f"{instruction}<|endofprompt|>{transcription}"


def apply_torch_compile(cosyvoice: FastCosyVoice3) -> None:
    """
    Applies torch.compile to LLM model for inference acceleration.
    
    Compiles the internal Qwen2ForCausalLM.model (Qwen2Model),
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
    cosyvoice: FastCosyVoice3,
    prompt_text: str,
    spk_id: str,
) -> None:
    """
    Warms up the model by generating tokens to compile all execution paths.
    
    torch.compile creates different kernels for different input sizes,
    so the model needs to be warmed up on texts of different lengths.
    
    Args:
        cosyvoice: Initialized FastCosyVoice3 model
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
        ):
            pass  # Just generate all segments
    
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
        ):
            pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    warmup_time = time.time() - warmup_start
    logger.info(f"Warmup completed in {warmup_time:.2f} sec")


def synthesize(
    cosyvoice: FastCosyVoice3,
    text: str,
    prompt_text: str,
    spk_id: str,
    sample_rate: int,
    output_path: str,
    speed: float = 1.0
) -> dict:
    """
    Performs non-streaming synthesis of text and returns metrics.
    
    Args:
        cosyvoice: FastCosyVoice3 model
        text: Text for synthesis
        prompt_text: Reference audio transcription
        spk_id: Speaker ID
        sample_rate: Sample rate
        output_path: Path to save the result
        speed: Speech speed multiplier (1.0 = normal)
    
    Returns:
        dict with keys: total_time, audio_duration, rtf, segment_count
    """
    start_time = time.time()
    audio_segments: list[bytes] = []
    segment_count = 0

    infer_ctx = torch.inference_mode() if USE_INFERENCE_MODE else torch.no_grad()
    with infer_ctx:
        for pcm_bytes in cosyvoice.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
            speed=speed,
        ):
            segment_count += 1
            audio_segments.append(pcm_bytes)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_time = time.time() - start_time
    
    # Concatenate segments and save as WAV
    if audio_segments:
        full_pcm = b''.join(audio_segments)
        # Save as WAV (PCM int16, mono)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(full_pcm)
        # Calculate duration from PCM bytes (2 bytes per sample, mono)
        audio_duration = len(full_pcm) / 2 / sample_rate
    else:
        audio_duration = 0.0
    
    rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
    
    return {
        'total_time': total_time,
        'audio_duration': audio_duration,
        'rtf': rtf,
        'segment_count': segment_count,
    }


def main():
    print("=" * 70)
    print("FastCosyVoice3 TTS - Non-streaming (Offline) Inference")
    print("=" * 70)
    
    # Check for model / reference
    try:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Model not found: {MODEL_DIR}")
        if not os.path.exists(REFERENCE_AUDIO):
            raise FileNotFoundError(f"Reference audio not found: {REFERENCE_AUDIO}")
    except FileNotFoundError as e:
        logger.error(str(e), exc_info=True)
        return
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load prompt_text from txt file next to audio
    prompt_text = load_prompt_text(REFERENCE_AUDIO, INSTRUCTION)
    
    print(f"\n🎤 Reference audio: {REFERENCE_AUDIO}")
    print(f"📝 Text for synthesis: {SYNTHESIS_TEXT[:80]}{'...' if len(SYNTHESIS_TEXT) > 80 else ''}")
    
    # Load model with parallel pipeline and TensorRT
    print("\n🔧 Loading FastCosyVoice3...")
    print(f"   - TensorRT Flow: {'✅' if USE_TRT_FLOW else '❌'}")
    print(f"   - TensorRT-LLM:  {'✅' if USE_TRT_LLM else '❌'} (dtype={TRT_LLM_DTYPE})")
    
    load_start = time.time()
    
    cosyvoice = FastCosyVoice3(
        model_dir=MODEL_DIR,
        fp16=True,
        load_trt=USE_TRT_FLOW,       # TensorRT for Flow decoder (~2.5x speedup)
        load_trt_llm=USE_TRT_LLM,    # TensorRT-LLM for LLM (~3x speedup)
        trt_llm_dtype=TRT_LLM_DTYPE,
        trt_llm_kv_cache_fraction=TRT_LLM_KV_CACHE_FRACTION,
    )
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f} sec")
    
    if USE_TRT_LLM and cosyvoice.trt_llm_loaded:
        print("✅ TensorRT-LLM loaded successfully")
    elif USE_TRT_LLM:
        print("⚠️ TensorRT-LLM not loaded, using PyTorch")
    
    print_gpu_memory("after model loading")
    
    # dtype diagnostics
    llm_dtype = next(cosyvoice.model.llm.parameters()).dtype
    flow_dtype = next(cosyvoice.model.flow.parameters()).dtype
    hift_dtype = next(cosyvoice.model.hift.parameters()).dtype
    print(f"📊 LLM dtype: {llm_dtype}, Flow dtype: {flow_dtype}, HiFT dtype: {hift_dtype}")
    
    sample_rate = cosyvoice.sample_rate
    print(f"📊 Sample rate: {sample_rate} Hz")
    
    # Parallel pipeline information
    print("\n🚀 Inference mode: Non-streaming (offline)")
    if USE_TRT_LLM and cosyvoice.trt_llm_loaded:
        print("   - LLM: TensorRT-LLM (~3x speedup)")
    else:
        print("   - LLM: PyTorch + torch.compile")
    if USE_TRT_FLOW:
        print("   - Flow: TensorRT (~2.5x speedup)")
    else:
        print("   - Flow: PyTorch")
    print("   - Hift: PyTorch (f0_predictor on CPU)")
    
    # Apply torch.compile to LLM only if TRT-LLM is not used
    if not (USE_TRT_LLM and cosyvoice.trt_llm_loaded):
        print("\n⚡ Applying torch.compile to LLM...")
        compile_start = time.time()
        apply_torch_compile(cosyvoice)
        compile_time = time.time() - compile_start
        print(f"✅ torch.compile applied in {compile_time:.3f} sec")
    else:
        print("\n⚡ torch.compile skipped (using TensorRT-LLM)")
    
    # Prepare speaker embeddings (once)
    print("\n🎯 Preparing speaker embeddings...")
    spk_id = "reference_speaker"
    embed_start = time.time()
    cosyvoice.add_zero_shot_spk(prompt_text, REFERENCE_AUDIO, spk_id)
    embed_time = time.time() - embed_start
    print(f"✅ Embeddings prepared in {embed_time:.3f} sec")
    
    # Model warmup
    if USE_TRT_LLM and cosyvoice.trt_llm_loaded:
        # With TRT-LLM warmup is shorter - only Flow and Hift
        print("\n🔥 Warming up model (TRT-LLM doesn't require long warmup)...")
        for _ in cosyvoice.inference_zero_shot(
            tts_text="Short model warmup.",
            prompt_text=prompt_text,
            prompt_wav=REFERENCE_AUDIO,
            zero_shot_spk_id=spk_id,
        ):
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("✅ Model warmed up")
    else:
        # Without TRT-LLM full warmup is needed for torch.compile
        print("\n🔥 Warming up model (compiling graphs for different text lengths)...")
        warmup_model(cosyvoice, prompt_text, spk_id)
        print("✅ Model warmed up and ready")
    
    print_gpu_memory("after warmup")
    
    # Reset peak memory counter to measure only generation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Generate text
    print("\n" + "=" * 70)
    print("📄 Generating audio")
    print("=" * 70)
    print(f"📝 {SYNTHESIS_TEXT[:80]}{'...' if len(SYNTHESIS_TEXT) > 80 else ''}")
    
    output_file = os.path.join(OUTPUT_DIR, 'output.wav')
    
    try:
        metrics = synthesize(
            cosyvoice=cosyvoice,
            text=SYNTHESIS_TEXT,
            prompt_text=prompt_text,
            spk_id=spk_id,
            sample_rate=sample_rate,
            output_path=output_file,
        )
        
        print(f"\n💾 Saved: {output_file}")
        print("\n📊 METRICS:")
        print("-" * 40)
        print(f"⏱️  Total time:       {metrics['total_time']:.3f} sec")
        print(f"🎵 Duration:         {metrics['audio_duration']:.3f} sec")
        print(f"📈 RTF:              {metrics['rtf']:.3f}")
        print(f"📦 Segments:         {metrics['segment_count']}")
        
        if metrics['rtf'] < 1.0:
            print(f"✅ Faster than real-time by {1/metrics['rtf']:.1f}x")
        else:
            print(f"⚠️  Slower than real-time by {metrics['rtf']:.1f}x")
        
        # Final summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY (FastCosyVoice3 - Non-streaming)")
        print("=" * 70)
        
        # Configuration
        llm_backend = "TensorRT-LLM" if (USE_TRT_LLM and cosyvoice.trt_llm_loaded) else "PyTorch+torch.compile"
        flow_backend = "TensorRT" if USE_TRT_FLOW else "PyTorch"
        print(f"LLM:  {llm_backend}")
        print(f"Flow: {flow_backend}")
        print("-" * 40)
        
        print(f"RTF:                 {metrics['rtf']:.3f}")
        print(f"Audio duration:      {metrics['audio_duration']:.3f} sec")
        print(f"Total time:          {metrics['total_time']:.3f} sec")
        
        if metrics['rtf'] < 1.0:
            print(f"\n✅ Speed: {1/metrics['rtf']:.1f}x faster than real-time")
            
    except Exception as e:
        logger.error(f"Error synthesizing text: {e}", exc_info=True)
        return
    
    # Attempt to free temporary PyTorch buffers
    # Important: KV-cache TensorRT-LLM and TensorRT workspace are not freed this way
    # (they live as long as the runner/engine lives).
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        logger.error(f"Error clearing memory: {e}", exc_info=True)
    
    print_gpu_memory("after generation")
    
    print("\n" + "=" * 70)
    print("✅ GENERATION COMPLETED!")
    print("=" * 70)
    print(f"\n📁 Results: {OUTPUT_DIR}/")
    
    print_gpu_memory("at the end")


if __name__ == '__main__':
    main()

