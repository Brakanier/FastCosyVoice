# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 FastCosyVoice Implementation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FastCosyVoice3 - High-level interface for parallel pipeline TTS.

This module provides the FastCosyVoice3 class which is the main entry point
for using the parallel pipeline CosyVoice3 model.

Supports TensorRT acceleration for both Flow decoder and LLM:
- Flow decoder TensorRT: ~2.5x speedup
- LLM TensorRT-LLM: ~3x speedup
"""

import os
import subprocess
import threading
import time
from typing import Generator

import torch
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download

from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.cli.model import CosyVoice3Model as OriginalCosyVoice3Model

from .frontend import CosyVoiceFrontEnd
from .model import FastCosyVoice3Model


class FastCosyVoice3:
    """
    FastCosyVoice3 - Parallel Pipeline TTS Interface.
    
    This class provides a high-level interface for CosyVoice3 with parallel
    pipeline processing. It supports only streaming zero-shot inference
    for maximum performance.
    
    Supports TensorRT acceleration for both Flow decoder and LLM.
    
    Example:
        >>> model = FastCosyVoice3(
        ...     "pretrained_models/CosyVoice3-0.5B", 
        ...     fp16=True, 
        ...     load_trt=True,      # Flow decoder TensorRT
        ...     load_trt_llm=True   # LLM TensorRT-LLM (~3x speedup)
        ... )
        >>> for chunk in model.inference_zero_shot_stream(
        ...     "Hello world",
        ...     "Reference text",
        ...     "reference.wav"
        ... ):
        ...     audio = chunk['tts_speech']
        ...     # Process audio chunk
    """
    
    def __init__(
        self,
        model_dir: str,
        fp16: bool = True,
        load_trt: bool = True,
        load_trt_llm: bool = False,
        trt_concurrent: int = 1,
        trt_llm_dtype: str = 'bfloat16',
        trt_llm_max_batch_size: int = 1,
        trt_llm_kv_cache_fraction: float = 0.3,
        flow_n_timesteps: int = 10,
    ):
        """
        Initialize FastCosyVoice3 with parallel pipeline.
        
        Args:
            model_dir: Path to model directory or ModelScope model ID
            fp16: Whether to use FP16 precision (recommended for speed)
            load_trt: Whether to load TensorRT for Flow decoder (highly recommended)
            load_trt_llm: Whether to load TensorRT-LLM for LLM (~3x speedup)
            trt_concurrent: Number of concurrent TRT contexts for Flow
            trt_llm_dtype: Data type for TRT-LLM (bfloat16, float16, float32)
            trt_llm_max_batch_size: Max batch size for TRT-LLM engine
            trt_llm_kv_cache_fraction: GPU memory fraction for KV cache
            flow_n_timesteps: Number of diffusion steps for Flow (10=best quality, 5-6=faster)
        """
        self.model_dir = model_dir
        self.fp16 = fp16
        self.trt_llm_loaded = False
        
        # Download model if not exists
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
            self.model_dir = model_dir
        
        # Load config
        hyper_yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(f'{hyper_yaml_path} not found!')
        
        # HuggingFace LLM directory (contains tokenizer and base weights)
        self.hf_llm_dir = os.path.join(model_dir, 'CosyVoice-BlankEN')
        
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(
                f, 
                overrides={'qwen_pretrain_path': self.hf_llm_dir}
            )
        
        # Verify model type
        model_type = get_model_type(configs)
        if model_type != OriginalCosyVoice3Model:
            raise ValueError(f'Expected CosyVoice3Model, got {model_type}. Use the original CosyVoice for other model types.')
        
        # Initialize frontend
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            os.path.join(model_dir, 'campplus.onnx'),
            os.path.join(model_dir, 'speech_tokenizer_v3.onnx'),
            os.path.join(model_dir, 'spk2info.pt'),
            configs['allowed_special']
        )
        
        self.sample_rate = configs['sample_rate']  # 24000
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            if fp16:
                fp16 = False
                logging.warning('No CUDA device, setting fp16 to False')
            if load_trt:
                load_trt = False
                logging.warning('No CUDA device, setting load_trt to False')
            if load_trt_llm:
                load_trt_llm = False
                logging.warning('No CUDA device, setting load_trt_llm to False')
        
        # Initialize parallel pipeline model
        self.model = FastCosyVoice3Model(
            configs['llm'],
            configs['flow'],
            configs['hift'],
            fp16
        )

        llm_pt_path = os.path.join(model_dir, 'llm.pt')
        flow_pt_path = os.path.join(model_dir, 'flow.pt')
        hift_pt_path = os.path.join(model_dir, 'hift.pt')

        # If TRT-LLM artifacts already exist, we can skip loading PyTorch LLM to GPU entirely.
        # This avoids a large, unnecessary VRAM allocation (TRT-LLM handles LLM inference).
        # NOTE: If artifacts are missing, _load_trt_llm may need PyTorch LLM weights to build hf_merged.
        skip_pytorch_llm = False
        if load_trt_llm and torch.cuda.is_available():
            hf_merged_dir = os.path.join(model_dir, 'hf_merged')
            trt_engines_dir = os.path.join(model_dir, f'trt_llm_engines_{trt_llm_dtype}_merged')
            metadata_path = os.path.join(hf_merged_dir, 'cosyvoice3_metadata.json')
            if os.path.exists(metadata_path) and os.path.isdir(trt_engines_dir):
                engine_files = [f for f in os.listdir(trt_engines_dir) if f.endswith('.engine')]
                if engine_files:
                    skip_pytorch_llm = True
                    logging.info('TRT-LLM artifacts found; skipping PyTorch LLM weight load to save VRAM')

        # Load weights (optionally skipping PyTorch LLM)
        self.model.load(
            llm_pt_path,
            flow_pt_path,
            hift_pt_path,
            load_llm=not skip_pytorch_llm,
        )
        
        # Load TensorRT for Flow decoder (significant speedup!)
        if load_trt:
            if fp16:
                logging.warning('DiT TensorRT fp16 engine may have performance issues, use with caution!')
            trt_model_path = os.path.join(
                model_dir, 
                f'flow.decoder.estimator.{"fp16" if fp16 else "fp32"}.mygpu.plan'
            )
            onnx_model_path = os.path.join(model_dir, 'flow.decoder.estimator.fp32.onnx')
            self.model.load_trt(trt_model_path, onnx_model_path, trt_concurrent, fp16)
        
        # Load TensorRT-LLM for LLM (~3x speedup!)
        if load_trt_llm:
            self._load_trt_llm(
                dtype=trt_llm_dtype,
                max_batch_size=trt_llm_max_batch_size,
                kv_cache_fraction=trt_llm_kv_cache_fraction,
            )
        
        del configs
        logging.info(f'FastCosyVoice3 initialized with fp16={fp16}, load_trt={load_trt}, load_trt_llm={load_trt_llm}')
    
    def _load_trt_llm(
        self,
        dtype: str = 'bfloat16',
        max_batch_size: int = 1,
        kv_cache_fraction: float = 0.3,
    ):
        """
        Load TensorRT-LLM for LLM inference.
        
        Uses merged HuggingFace model with speech_embedding and llm_decoder
        integrated into embed_tokens and lm_head respectively.
        
        Args:
            dtype: Data type (bfloat16, float16, float32)
            max_batch_size: Maximum batch size
            kv_cache_fraction: GPU memory fraction for KV cache
        """
        import json
        
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelRunnerCpp
            from transformers import AutoTokenizer
        except ImportError:
            logging.warning(
                'TensorRT-LLM not installed, using PyTorch LLM. '
                'Install with: pip install tensorrt-llm',
                exc_info=True
            )
            return
        
        # Paths for merged model
        hf_merged_dir = os.path.join(self.model_dir, 'hf_merged')
        trt_weights_dir = os.path.join(self.model_dir, f'trt_llm_weights_{dtype}_merged')
        trt_engines_dir = os.path.join(self.model_dir, f'trt_llm_engines_{dtype}_merged')
        metadata_path = os.path.join(hf_merged_dir, 'cosyvoice3_metadata.json')
        
        # Check if merged model exists
        if not os.path.exists(hf_merged_dir):
            logging.info('Merged HuggingFace model not found, creating...')
            self._create_merged_hf_model(hf_merged_dir, dtype)
        
        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.trt_llm_metadata = json.load(f)
        else:
            raise RuntimeError(
                f'Metadata file not found: {metadata_path}. '
                f'Run scripts/convert_cosyvoice3_to_hf.py first.'
            )
        
        self.speech_token_offset = self.trt_llm_metadata['speech_token_offset']
        # ВАЖНО: используем base_speech_token_size для валидного диапазона Flow
        # Для обратной совместимости поддерживаем старый ключ 'speech_token_size'
        self.speech_token_size = self.trt_llm_metadata.get(
            'base_speech_token_size', 
            self.trt_llm_metadata.get('speech_token_size', 6561)
        )
        
        logging.info(f'Valid speech token range for Flow: [0, {self.speech_token_size})')
        
        # Auto-convert to TRT-LLM if needed
        if not os.path.exists(trt_engines_dir) or not os.path.isdir(trt_engines_dir):
            logging.info('TRT-LLM engines not found, converting merged model...')
            self._convert_merged_to_trt(hf_merged_dir, trt_weights_dir, trt_engines_dir, dtype, max_batch_size)
        else:
            engine_files = [f for f in os.listdir(trt_engines_dir) if f.endswith('.engine')]
            if not engine_files:
                logging.info('TRT-LLM engine files not found, rebuilding...')
                self._convert_merged_to_trt(hf_merged_dir, trt_weights_dir, trt_engines_dir, dtype, max_batch_size)
        
        # Load tokenizer from merged model (must include speech tokens AND CosyVoice3 text special tokens like [cough])
        self.trt_llm_tokenizer = AutoTokenizer.from_pretrained(hf_merged_dir, trust_remote_code=True)
        try:
            unk_id = getattr(self.trt_llm_tokenizer, "unk_token_id", None)
            for tok in ("[cough]", "[laughter]"):
                tid = self.trt_llm_tokenizer.convert_tokens_to_ids(tok)
                if unk_id is not None and tid == unk_id:
                    logging.warning(
                        f"TRT-LLM tokenizer is missing text special token {tok}. "
                        f"This usually means {hf_merged_dir} was built with an old converter. "
                        f"Delete hf_merged and TRT engine dirs to force rebuild."
                    )
        except Exception:
            # Never fail load because of diagnostic checks
            logging.debug("Could not validate TRT tokenizer special tokens", exc_info=True)
        
        # In CosyVoice3, special tokens are INSIDE speech_embedding:
        # - sos = speech_token_size + 0 = 6561
        # - eos = speech_token_size + 1 = 6562
        # - task_id = speech_token_size + 2 = 6563
        # So we use <|s_6561|>, <|s_6562|>, <|s_6563|> for these
        base_speech_token_size = self.speech_token_size  # 6561 for valid Flow tokens
        
        # Real special token indices INSIDE speech_embedding
        self.sos_speech_idx = base_speech_token_size + 0  # 6561
        self.eos_speech_idx = base_speech_token_size + 1  # 6562  
        self.task_id_speech_idx = base_speech_token_size + 2  # 6563
        
        # Token IDs in merged vocab = speech_token_offset + speech_idx
        self.sos_token_id = self.speech_token_offset + self.sos_speech_idx
        self.eos1_token_id = self.speech_token_offset + self.eos_speech_idx
        self.task_id_token_id = self.speech_token_offset + self.task_id_speech_idx
        
        logging.info(f'Speech token offset: {self.speech_token_offset}')
        logging.info(f'Base speech token size (for Flow): {self.speech_token_size}')
        logging.info(f'Valid speech token range: [{self.speech_token_offset}, {self.speech_token_offset + self.speech_token_size})')
        logging.info(f'SOS token: <|s_{self.sos_speech_idx}|> = ID {self.sos_token_id}')
        logging.info(f'EOS token: <|s_{self.eos_speech_idx}|> = ID {self.eos1_token_id}')
        logging.info(f'Task ID token: <|s_{self.task_id_speech_idx}|> = ID {self.task_id_token_id}')
        
        # Initialize TRT-LLM runner
        runtime_rank = tensorrt_llm.mpi_rank()
        
        runner_kwargs = dict(
            engine_dir=trt_engines_dir,
            rank=runtime_rank,
            max_output_len=2048,
            enable_context_fmha_fp32_acc=False,
            max_batch_size=max_batch_size,
            max_input_len=4096,
            kv_cache_free_gpu_memory_fraction=kv_cache_fraction,
            cuda_graph_mode=False,
            gather_generation_logits=False,
        )
        
        self.trt_llm_runner = ModelRunnerCpp.from_dir(**runner_kwargs)
        self.trt_llm_loaded = True
        
        # Free PyTorch LLM layers to save VRAM
        try:
            del self.model.llm.llm.model.model.layers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info('Freed PyTorch LLM layers (using TRT-LLM)')
        except Exception as e:
            logging.warning(f'Could not free PyTorch LLM layers: {e}')
        
        logging.info(f'TRT-LLM loaded from {trt_engines_dir}')
    
    def _create_merged_hf_model(self, output_dir: str, dtype: str):
        """
        Create merged HuggingFace model with speech_embedding and llm_decoder.
        
        This merges:
        - speech_embedding -> embed_tokens[original_vocab_size:]
        - llm_decoder -> lm_head[original_vocab_size:]
        
        IMPORTANT: Speech tokens are added FIRST, then special tokens.
        This ensures speech_token_offset = original_vocab_size and
        special tokens (<|sos|>, <|eos1|>, etc.) are OUTSIDE the speech token range.
        """
        import json
        from transformers import AutoTokenizer
        
        logging.info('Creating merged HuggingFace model with speech tokens...')
        
        # Extract components from CosyVoice3 LLM
        qwen_model = self.model.llm.llm.model  # Qwen2ForCausalLM
        speech_embedding = self.model.llm.speech_embedding
        llm_decoder = self.model.llm.llm_decoder
        
        # ВАЖНО: Различаем два размера:
        # - base_speech_token_size: реальные speech токены для Flow (0 до N-1)
        # - embedding_size: включает специальные токены (sos, eos, task_id, fill, +200)
        base_speech_token_size = self.model.llm.speech_token_size  # Реальный размер для Flow
        embedding_size = speech_embedding.num_embeddings  # Полный размер с спец. токенами
        
        logging.info(f'Base speech token size (for Flow): {base_speech_token_size}')
        logging.info(f'Embedding size (with special tokens): {embedding_size}')
        
        def _add_cosyvoice3_text_special_tokens(tok):
            """
            Ensure CosyVoice3 text special tokens (e.g. [cough], [laughter]) exist in tokenizer.
            This mirrors `CosyVoice3Tokenizer` behavior from `cosyvoice/tokenizer/tokenizer.py`.
            """
            special_tokens = {
                'eos_token': '<|endoftext|>',
                'pad_token': '<|endoftext|>',
                'additional_special_tokens': [
                    '<|im_start|>', '<|im_end|>', '<|endofprompt|>',
                    '[breath]', '<strong>', '</strong>', '[noise]',
                    '[laughter]', '[cough]', '[clucking]', '[accent]',
                    '[quick_breath]',
                    "<laughter>", "</laughter>",
                    "[hissing]", "[sigh]", "[vocalized-noise]",
                    "[lipsmack]", "[mn]", "<|endofsystem|>",
                    # Phoneme tokens (kept in sync with CosyVoice3Tokenizer)
                    "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]", "[AH]", "[AH0]", "[AH1]", "[AH2]",
                    "[AO]", "[AO0]", "[AO1]", "[AO2]", "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
                    "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]", "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]",
                    "[EY0]", "[EY1]", "[EY2]", "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]", "[IY]", "[IY0]", "[IY1]",
                    "[IY2]", "[JH]", "[K]", "[L]", "[M]", "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
                    "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]", "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]",
                    "[UW0]", "[UW1]", "[UW2]", "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
                    "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]", "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]",
                    "[g]", "[h]", "[i]", "[ian]", "[in]", "[ing]", "[iu]", "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]",
                    "[iáng]", "[iáo]", "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]", "[iā]", "[iān]", "[iāng]", "[iāo]",
                    "[iē]", "[iě]", "[iōng]", "[iū]", "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]", "[j]", "[k]", "[l]",
                    "[m]", "[n]", "[o]", "[ong]", "[ou]", "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]", "[u]", "[uang]", "[ue]",
                    "[un]", "[uo]", "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]", "[uè]", "[ué]", "[uì]",
                    "[uí]", "[uò]", "[uó]", "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]", "[uǎ]", "[uǎi]",
                    "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]", "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]", "[à]", "[ài]", "[àn]", "[àng]",
                    "[ào]", "[á]", "[ái]", "[án]", "[áng]", "[áo]", "[è]", "[èi]", "[èn]", "[èng]", "[èr]", "[é]", "[éi]", "[én]",
                    "[éng]", "[ér]", "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]", "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
                    "[ù]", "[ùn]", "[ú]", "[ún]", "[ā]", "[āi]", "[ān]", "[āng]", "[āo]", "[ē]", "[ēi]", "[ēn]", "[ēng]", "[ě]",
                    "[ěi]", "[ěn]", "[ěng]", "[ěr]", "[ī]", "[īn]", "[īng]", "[ō]", "[ōng]", "[ōu]", "[ū]", "[ūn]", "[ǎ]", "[ǎi]",
                    "[ǎn]", "[ǎng]", "[ǎo]", "[ǐ]", "[ǐn]", "[ǐng]", "[ǒ]", "[ǒng]", "[ǒu]", "[ǔ]", "[ǔn]", "[ǘ]", "[ǚ]", "[ǜ]",
                ],
            }
            tok.add_special_tokens(special_tokens)
            return tok

        # Load and extend tokenizer (MUST include CosyVoice3 text special tokens like [cough])
        tokenizer = AutoTokenizer.from_pretrained(self.hf_llm_dir, trust_remote_code=True)
        base_vocab_size = len(tokenizer)
        tokenizer = _add_cosyvoice3_text_special_tokens(tokenizer)
        text_vocab_size = len(tokenizer)
        logging.info(f'Base vocab size: {base_vocab_size}, after text special tokens: {text_vocab_size}')
        
        # In CosyVoice3, special tokens are INSIDE speech_embedding:
        # - sos = base_speech_token_size + 0 = 6561
        # - eos = base_speech_token_size + 1 = 6562
        # - task_id = base_speech_token_size + 2 = 6563
        # - fill = base_speech_token_size + 3 = 6564
        # So we DON'T need separate <|sos|>, <|eos1|> tokens - they're <|s_6561|>, <|s_6562|>, etc.
        
        # Add speech tokens (they include speech special tokens as <|s_6561|>, <|s_6562|>, etc.)
        speech_tokens = [f"<|s_{i}|>" for i in range(embedding_size)]
        tokenizer.add_tokens(speech_tokens)
        speech_token_offset = text_vocab_size  # <|s_0|> has ID = text_vocab_size (after text special tokens)
        
        new_vocab_size = len(tokenizer)
        padded_vocab_size = ((new_vocab_size + 127) // 128) * 128
        
        # Special tokens are inside speech_embedding
        sos_speech_idx = base_speech_token_size + 0  # 6561
        eos_speech_idx = base_speech_token_size + 1  # 6562
        task_id_speech_idx = base_speech_token_size + 2  # 6563
        
        sos_token_id = speech_token_offset + sos_speech_idx
        eos_token_id = speech_token_offset + eos_speech_idx
        task_id_token_id = speech_token_offset + task_id_speech_idx
        
        logging.info(f'Speech token range: [{speech_token_offset}, {speech_token_offset + embedding_size})')
        logging.info(f'SOS: <|s_{sos_speech_idx}|> = ID {sos_token_id}')
        logging.info(f'EOS: <|s_{eos_speech_idx}|> = ID {eos_token_id}')
        logging.info(f'Task ID: <|s_{task_id_speech_idx}|> = ID {task_id_token_id}')
        
        # Resize embeddings
        qwen_model.resize_token_embeddings(padded_vocab_size)
        input_embeddings = qwen_model.get_input_embeddings()
        hidden_size = input_embeddings.weight.shape[1]
        
        # Copy speech_embedding to extended embed_tokens
        # Speech tokens are at [original_vocab_size, original_vocab_size + embedding_size)
        with torch.no_grad():
            input_embeddings.weight[speech_token_offset:speech_token_offset + embedding_size] = \
                speech_embedding.weight[:embedding_size].to(input_embeddings.weight.dtype)
        
        # Create new lm_head with llm_decoder
        has_bias = llm_decoder.bias is not None
        new_lm_head = torch.nn.Linear(hidden_size, padded_vocab_size, bias=has_bias)
        
        with torch.no_grad():
            new_lm_head.weight.data.zero_()
            if has_bias:
                new_lm_head.bias.data.fill_(-float('inf'))
            
            # Copy original lm_head for text tokens (including any text special tokens already present in qwen_model)
            original_lm_head = qwen_model.lm_head
            if original_lm_head is not None:
                copy_size = min(original_lm_head.weight.shape[0], text_vocab_size)
                new_lm_head.weight[:copy_size] = original_lm_head.weight[:copy_size]
                if has_bias and original_lm_head.bias is not None:
                    new_lm_head.bias[:copy_size] = original_lm_head.bias[:copy_size]
            
            # Copy llm_decoder for speech tokens (полный размер)
            decoder_size = llm_decoder.weight.shape[0]
            new_lm_head.weight[speech_token_offset:speech_token_offset + decoder_size] = \
                llm_decoder.weight[:decoder_size].to(new_lm_head.weight.dtype)
            if has_bias:
                new_lm_head.bias[speech_token_offset:speech_token_offset + decoder_size] = \
                    llm_decoder.bias[:decoder_size].to(new_lm_head.bias.dtype)
        
        qwen_model.lm_head = new_lm_head
        
        # Update config
        qwen_model.config.vocab_size = padded_vocab_size
        qwen_model.config.tie_word_embeddings = False
        
        qwen_model.config.eos_token_id = eos_token_id
        qwen_model.generation_config.eos_token_id = eos_token_id
        qwen_model.generation_config.pad_token_id = eos_token_id
        
        # Convert dtype
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        qwen_model.to(dtype_map.get(dtype, torch.bfloat16))
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        qwen_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save metadata
        metadata = {
            "original_vocab_size": base_vocab_size,
            "text_vocab_size": text_vocab_size,
            "base_speech_token_size": base_speech_token_size,  # Для Flow (реальные speech токены 0-6560)
            "embedding_size": embedding_size,  # Полный размер с спец. токенами
            "padded_vocab_size": padded_vocab_size,
            "speech_token_offset": speech_token_offset,
            # Special token indices INSIDE speech_embedding
            "sos_speech_idx": sos_speech_idx,
            "eos_speech_idx": eos_speech_idx,
            "task_id_speech_idx": task_id_speech_idx,
            # Corresponding token IDs in merged vocab
            "sos_token_id": sos_token_id,
            "eos_token_id": eos_token_id,
            "task_id_token_id": task_id_token_id,
            "dtype": dtype,
        }
        with open(os.path.join(output_dir, "cosyvoice3_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Free qwen_model from GPU memory after saving
        del qwen_model, tokenizer, new_lm_head, input_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info(f'Merged HuggingFace model saved to {output_dir}')
        logging.info(f'Valid speech token range for Flow: [0, {base_speech_token_size})')
    
    def _convert_merged_to_trt(
        self,
        hf_dir: str,
        trt_weights_dir: str,
        trt_engines_dir: str,
        dtype: str,
        max_batch_size: int,
    ):
        """
        Convert merged HuggingFace model to TensorRT-LLM.
        """
        os.makedirs(trt_weights_dir, exist_ok=True)
        os.makedirs(trt_engines_dir, exist_ok=True)
        
        logging.info('Step 1/2: Converting merged model to TRT-LLM checkpoints...')
        
        try:
            from tensorrt_llm.models import QWenForCausalLM
            from tensorrt_llm.mapping import Mapping
            from tensorrt_llm.models.modeling_utils import QuantConfig
            
            mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
            
            qwen = QWenForCausalLM.from_hugging_face(
                hf_dir, dtype, mapping=mapping, quant_config=QuantConfig()
            )
            qwen.save_checkpoint(trt_weights_dir, save_config=True)
            del qwen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info(f'Checkpoint converted to {trt_weights_dir}')
            
        except Exception as e:
            logging.error(f'Failed to convert checkpoint: {e}', exc_info=True)
            raise RuntimeError(f'Failed to convert merged model to TRT-LLM: {e}')
        
        logging.info('Step 2/2: Building TRT-LLM engines...')
        
        try:
            build_cmd = [
                'trtllm-build',
                '--checkpoint_dir', trt_weights_dir,
                '--output_dir', trt_engines_dir,
                '--max_batch_size', str(max_batch_size),
                '--max_input_len', '4096',
                '--max_num_tokens', '16384',
                '--gemm_plugin', dtype,
            ]
            
            result = subprocess.run(build_cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode != 0:
                raise RuntimeError(f'trtllm-build failed: {result.stderr}')
            
            logging.info(f'TRT-LLM engines built at {trt_engines_dir}')
            
        except subprocess.TimeoutExpired:
            raise RuntimeError('TRT-LLM engine build timed out')
        except FileNotFoundError:
            raise RuntimeError('trtllm-build command not found')
        except Exception as e:
            logging.error(f'Failed to build TRT-LLM engines: {e}', exc_info=True)
            raise
    
    def _extract_speech_ids(self, generated_ids: list) -> list:
        """
        Extract speech token IDs from generated token IDs.
        
        In the merged model, speech tokens are at indices:
        [speech_token_offset, speech_token_offset + speech_token_size)
        
        The actual speech token ID is: token_id - speech_token_offset
        """
        speech_ids = []
        
        for tid in generated_ids:
            # Check if this is a speech token (in the range [offset, offset + size))
            if self.speech_token_offset <= tid < self.speech_token_offset + self.speech_token_size:
                # Convert to actual speech token ID
                speech_id = tid - self.speech_token_offset
                speech_ids.append(speech_id)
            # Skip text tokens and special tokens outside speech range
        
        return speech_ids
    
    def _run_trt_llm_inference_streaming(
        self,
        text: str,
        prompt_text: str,
        prompt_speech_tokens: list,
        sampling: int = 25,
    ) -> Generator[int, None, None]:
        """
        Run LLM inference using TRT-LLM with TRUE STREAMING.
        
        CosyVoice3 LLM input structure:
        [sos_emb, text_embeddings, task_id_emb, prompt_speech_embeddings]
        
        Where:
        - sos = speech_token_size + 0 = <|s_6561|>
        - task_id = speech_token_size + 2 = <|s_6563|>
        - eos = speech_token_size + 1 = <|s_6562|> (for stopping)
        
        Yields speech tokens one by one AS THEY ARE GENERATED (in range [0, speech_token_size)).
        """
        # Build input prompt using correct special tokens
        full_text = prompt_text + text
        
        # Convert prompt speech tokens to string format
        prompt_speech_str = ''.join([f'<|s_{t}|>' for t in prompt_speech_tokens])
        
        # Build full prompt with CORRECT special tokens:
        # <|s_6561|> = sos, <|s_6563|> = task_id
        sos_token = f'<|s_{self.sos_speech_idx}|>'
        task_id_token = f'<|s_{self.task_id_speech_idx}|>'
        prompt = f"{sos_token}{full_text}{task_id_token}{prompt_speech_str}"
        
        # Tokenize
        input_ids = self.trt_llm_tokenizer.encode(prompt)
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.int32)]
        input_length = len(input_ids)
        
        # Estimate max tokens based on text length
        # ~25 speech tokens per second, ~10 chars per second in Russian/English
        # Add some margin for safety
        estimated_duration_sec = max(5, len(text) / 5)  # ~5 chars per second, min 5 seconds
        max_speech_tokens = int(estimated_duration_sec * 25 * 1.5)  # 1.5x margin
        max_new_tokens = min(max_speech_tokens + 100, 2048)  # Cap at 2048
        
        logging.debug(f'TRT-LLM: text_len={len(text)}, estimated_duration={estimated_duration_sec:.1f}s, max_new_tokens={max_new_tokens}')
        
        # Generate with STREAMING
        llm_gen_start = time.time()
        prev_output_len = 0
        total_raw_tokens = 0
        total_speech_tokens = 0
        
        try:
            with torch.inference_mode():
                # streaming=True returns an iterator that yields incremental outputs
                outputs_iter = self.trt_llm_runner.generate(
                    batch_input_ids=batch_input_ids,
                    max_new_tokens=max_new_tokens,
                    end_id=self.eos1_token_id,
                    pad_id=self.eos1_token_id,
                    temperature=0.8,
                    top_k=sampling,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    streaming=True,  # TRUE STREAMING!
                    output_sequence_lengths=True,
                    output_generation_logits=False,
                    return_dict=True,
                    return_all_generated_tokens=True  # Get all tokens each iteration
                )
                
                # Iterate over streaming outputs
                for outputs in outputs_iter:
                    output_ids = outputs["output_ids"]
                    sequence_lengths = outputs["sequence_lengths"]
                    
                    output_end = sequence_lengths[0][0].item()
                    current_generated = output_ids[0][0][input_length:output_end].tolist()
                    
                    # Extract only NEW tokens since last iteration
                    new_tokens = current_generated[prev_output_len:]
                    prev_output_len = len(current_generated)
                    total_raw_tokens += len(new_tokens)
                    
                    # Extract and yield speech tokens immediately
                    for tid in new_tokens:
                        if self.speech_token_offset <= tid < self.speech_token_offset + self.speech_token_size:
                            speech_id = tid - self.speech_token_offset
                            total_speech_tokens += 1
                            yield speech_id
        finally:
            # Cleanup batch input tensors
            del batch_input_ids
        
        llm_gen_elapsed = time.time() - llm_gen_start
        raw_tps = total_raw_tokens / llm_gen_elapsed if llm_gen_elapsed > 0 else 0
        speech_tps = total_speech_tokens / llm_gen_elapsed if llm_gen_elapsed > 0 else 0
        logging.info(
            f'TRT-LLM streaming: {total_raw_tokens} raw tokens ({raw_tps:.1f}/s), '
            f'{total_speech_tokens} speech tokens ({speech_tps:.1f}/s) in {llm_gen_elapsed:.3f}s'
        )
    
    def _trt_llm_job(
        self,
        text: str,
        prompt_text: str,
        prompt_speech_tokens: list,
        tokens_list: list,
        llm_end_flag: dict,
        tokens_lock,
        sampling: int = 25,
    ):
        """
        TRT-LLM token generation job - runs in dedicated thread.
        Populates tokens_list with speech tokens as they're generated.
        """
        llm_start_time = time.time()
        token_count = 0
        
        try:
            for speech_token in self._run_trt_llm_inference_streaming(
                text=text,
                prompt_text=prompt_text,
                prompt_speech_tokens=prompt_speech_tokens,
                sampling=sampling,
            ):
                with tokens_lock:
                    tokens_list.append(speech_token)
                token_count += 1
            
            llm_duration = time.time() - llm_start_time
            tokens_per_sec = token_count / llm_duration if llm_duration > 0 else 0
            logging.info(
                f'[TRT-LLM] duration={llm_duration:.3f}s, tokens={token_count}, tokens/s={tokens_per_sec:.2f}'
            )
            
        except Exception as e:
            logging.error(f'[TRT-LLM] Error: {e}', exc_info=True)
        finally:
            llm_end_flag['done'] = True
    
    def list_available_spks(self):
        """
        List available speaker IDs.
        
        Returns:
            List of speaker IDs that can be used with zero_shot_spk_id
        """
        return list(self.frontend.spk2info.keys())
    
    def add_zero_shot_spk(
        self,
        prompt_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str
    ) -> bool:
        """
        Add a new zero-shot speaker from prompt audio.
        
        Args:
            prompt_text: Text content of the prompt audio
            prompt_wav: Path to prompt audio file
            zero_shot_spk_id: ID to assign to this speaker
        
        Returns:
            True if successful
        """
        if zero_shot_spk_id == '':
            raise ValueError('zero_shot_spk_id cannot be empty')
        
        model_input = self.frontend.frontend_zero_shot(
            '', prompt_text, prompt_wav, self.sample_rate, ''
        )
        del model_input['text']
        del model_input['text_len']
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        return True
    
    def save_spkinfo(self):
        """Save speaker info to disk."""
        torch.save(
            self.frontend.spk2info, 
            os.path.join(self.model_dir, 'spk2info.pt')
        )
    
    def _tensor_to_pcm_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """
        Convert audio tensor to raw PCM int16 bytes.
        
        Args:
            audio_tensor: Audio tensor with shape [1, audio_len] or [audio_len], float values in [-1, 1]
        
        Returns:
            Raw PCM bytes (int16, little-endian)
        """
        # Ensure 1D tensor
        audio = audio_tensor.squeeze()
        
        # Clamp to [-1, 1] and convert to int16
        audio = audio.clamp(-1.0, 1.0)
        audio_int16 = (audio * 32767).to(torch.int16)
        
        # Convert to bytes (CPU, numpy, then tobytes)
        return audio_int16.cpu().numpy().tobytes()
    
    def inference_zero_shot_stream(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str = '',
        text_frontend: bool = True
    ) -> Generator[bytes, None, None]:
        """
        Zero-shot streaming TTS inference with parallel pipeline.
        
        This method runs LLM, Flow, and Hift in parallel for maximum performance.
        Audio chunks are yielded as soon as they're ready.
        
        If TRT-LLM is loaded, uses TensorRT-LLM for LLM inference (~3x faster).
        
        Args:
            tts_text: Text to synthesize
            prompt_text: Text content of the prompt audio
            prompt_wav: Path to prompt audio file
            zero_shot_spk_id: Optional speaker ID (if already registered)
            text_frontend: Whether to apply text normalization
        
        Yields:
            Raw PCM bytes (int16, little-endian, mono, sample_rate from model)
        """
        # Normalize prompt text
        prompt_text = self.frontend.text_normalize(
            prompt_text, split=False, text_frontend=text_frontend
        )
        
        # Process each text chunk
        for text_chunk in tqdm(
            self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend),
            desc='Synthesizing'
        ):
            # Warn if text is too short
            if not hasattr(text_chunk, '__iter__') or isinstance(text_chunk, str):
                if len(text_chunk) < 0.5 * len(prompt_text):
                    logging.warning(
                        f'Synthesis text "{text_chunk}" is shorter than prompt text, '
                        'this may lead to poor quality'
                    )
            
            # Prepare model input
            model_input = self.frontend.frontend_zero_shot(
                text_chunk, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            
            start_time = time.time()
            logging.info(f'Synthesizing: {text_chunk}')
            
            # Use TRT-LLM if available
            if self.trt_llm_loaded:
                # TRUE STREAMING: TRT-LLM generates tokens in a thread,
                # Flow+Hift processes them in parallel in another thread
                prompt_speech_tokens = model_input['llm_prompt_speech_token'].squeeze(0).tolist()
                
                # Shared state for threading
                tokens_list: list = []
                tokens_lock = threading.Lock()
                llm_end_flag = {'done': False}
                
                # Start TRT-LLM thread
                llm_thread = threading.Thread(
                    target=self._trt_llm_job,
                    args=(text_chunk, prompt_text, prompt_speech_tokens,
                          tokens_list, llm_end_flag, tokens_lock),
                    daemon=True
                )
                llm_thread.start()
                
                # Use streaming architecture: Flow+Hift runs in thread, consumes tokens as they arrive
                for model_output in self.model.tts_stream_external_llm(
                    tokens_list=tokens_list,
                    tokens_lock=tokens_lock,
                    llm_end_flag=llm_end_flag,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                ):
                    audio_tensor = model_output['tts_speech']
                    speech_len = audio_tensor.shape[1] / self.sample_rate
                    elapsed = time.time() - start_time
                    rtf = elapsed / speech_len if speech_len > 0 else 0
                    logging.info(f'Yield speech len={speech_len:.3f}s, rtf={rtf:.3f}')
                    yield self._tensor_to_pcm_bytes(audio_tensor)
                    start_time = time.time()
                
                # Wait for LLM thread to finish
                llm_thread.join(timeout=5.0)
            else:
                # Use PyTorch LLM with parallel pipeline
                for model_output in self.model.tts_stream(**model_input):
                    audio_tensor = model_output['tts_speech']
                    speech_len = audio_tensor.shape[1] / self.sample_rate
                    elapsed = time.time() - start_time
                    rtf = elapsed / speech_len if speech_len > 0 else 0
                    logging.info(f'Yield speech len={speech_len:.3f}s, rtf={rtf:.3f}')
                    yield self._tensor_to_pcm_bytes(audio_tensor)
                    start_time = time.time()
            
            # Note: model_input tensors freed automatically when out of scope
    
    def _run_trt_llm_inference(
        self,
        text: str,
        prompt_text: str,
        prompt_speech_tokens: list,
        sampling: int = 25,
    ) -> list:
        """
        Run LLM inference using TRT-LLM (non-streaming).
        
        Returns all speech tokens at once after generation completes.
        
        Args:
            text: Text to synthesize
            prompt_text: Prompt text
            prompt_speech_tokens: Prompt speech tokens
            sampling: Top-k sampling parameter
        
        Returns:
            List of speech tokens
        """
        # Build input prompt using correct special tokens
        full_text = prompt_text + text
        
        # Convert prompt speech tokens to string format
        prompt_speech_str = ''.join([f'<|s_{t}|>' for t in prompt_speech_tokens])
        
        # Build full prompt with CORRECT special tokens:
        # <|s_6561|> = sos, <|s_6563|> = task_id
        sos_token = f'<|s_{self.sos_speech_idx}|>'
        task_id_token = f'<|s_{self.task_id_speech_idx}|>'
        prompt = f"{sos_token}{full_text}{task_id_token}{prompt_speech_str}"
        
        # Tokenize
        input_ids = self.trt_llm_tokenizer.encode(prompt)
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.int32)]
        input_length = len(input_ids)
        
        # Estimate max tokens based on text length
        estimated_duration_sec = max(5, len(text) / 5)
        max_speech_tokens = int(estimated_duration_sec * 25 * 1.5)
        max_new_tokens = min(max_speech_tokens + 100, 2048)
        
        logging.debug(f'TRT-LLM: text_len={len(text)}, estimated_duration={estimated_duration_sec:.1f}s, max_new_tokens={max_new_tokens}')
        
        # Generate (non-streaming)
        llm_gen_start = time.time()
        speech_tokens = []
        
        try:
            with torch.inference_mode():
                outputs = self.trt_llm_runner.generate(
                    batch_input_ids=batch_input_ids,
                    max_new_tokens=max_new_tokens,
                    end_id=self.eos1_token_id,
                    pad_id=self.eos1_token_id,
                    temperature=0.8,
                    top_k=sampling,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    streaming=False,
                    output_sequence_lengths=True,
                    output_generation_logits=False,
                    return_dict=True,
                )
                
                output_ids = outputs["output_ids"]
                sequence_lengths = outputs["sequence_lengths"]
                
                output_end = sequence_lengths[0][0].item()
                generated_ids = output_ids[0][0][input_length:output_end].tolist()
                
                # Extract speech tokens
                speech_tokens = self._extract_speech_ids(generated_ids)
        finally:
            del batch_input_ids
        
        llm_gen_elapsed = time.time() - llm_gen_start
        raw_tps = len(generated_ids) / llm_gen_elapsed if llm_gen_elapsed > 0 else 0
        speech_tps = len(speech_tokens) / llm_gen_elapsed if llm_gen_elapsed > 0 else 0
        logging.info(
            f'TRT-LLM non-streaming: {len(generated_ids)} raw tokens ({raw_tps:.1f}/s), '
            f'{len(speech_tokens)} speech tokens ({speech_tps:.1f}/s) in {llm_gen_elapsed:.3f}s'
        )
        
        return speech_tokens
    
    def inference_zero_shot(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str = '',
        text_frontend: bool = True,
        speed: float = 1.0
    ) -> Generator[bytes, None, None]:
        """
        Zero-shot non-streaming TTS inference.
        
        This method generates all speech tokens first, then converts them to audio
        in one pass. This has higher latency to first audio but can be simpler
        for batch processing.
        
        If TRT-LLM is loaded, uses TensorRT-LLM for LLM inference (~3x faster).
        
        Args:
            tts_text: Text to synthesize
            prompt_text: Text content of the prompt audio
            prompt_wav: Path to prompt audio file
            zero_shot_spk_id: Optional speaker ID (if already registered)
            text_frontend: Whether to apply text normalization
            speed: Speech speed multiplier (1.0 = normal)
        
        Yields:
            Raw PCM bytes (int16, little-endian, mono, sample_rate from model)
            (yields one chunk per text segment after normalization/splitting)
        """
        # Normalize prompt text
        prompt_text = self.frontend.text_normalize(
            prompt_text, split=False, text_frontend=text_frontend
        )
        
        # Process each text chunk
        for text_chunk in tqdm(
            self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend),
            desc='Synthesizing'
        ):
            # Warn if text is too short
            if not hasattr(text_chunk, '__iter__') or isinstance(text_chunk, str):
                if len(text_chunk) < 0.5 * len(prompt_text):
                    logging.warning(
                        f'Synthesis text "{text_chunk}" is shorter than prompt text, '
                        'this may lead to poor quality'
                    )
            
            # Prepare model input
            model_input = self.frontend.frontend_zero_shot(
                text_chunk, prompt_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            
            start_time = time.time()
            logging.info(f'Synthesizing: {text_chunk}')
            
            # Use TRT-LLM if available
            if self.trt_llm_loaded:
                prompt_speech_tokens = model_input['llm_prompt_speech_token'].squeeze(0).tolist()
                
                # Generate all tokens (non-streaming)
                speech_tokens = self._run_trt_llm_inference(
                    text=text_chunk,
                    prompt_text=prompt_text,
                    prompt_speech_tokens=prompt_speech_tokens,
                )
                
                # Convert tokens to audio in one pass
                model_output = self.model.tts_with_external_tokens(
                    tokens=speech_tokens,
                    speed=speed,
                    **{k: v for k, v in model_input.items() if k.startswith('flow') or k.startswith('prompt_speech')}
                )
                
                audio_tensor = model_output['tts_speech']
                speech_len = audio_tensor.shape[1] / self.sample_rate
                elapsed = time.time() - start_time
                rtf = elapsed / speech_len if speech_len > 0 else 0
                logging.info(f'Generated speech len={speech_len:.3f}s, rtf={rtf:.3f}')
                yield self._tensor_to_pcm_bytes(audio_tensor)
            else:
                # Use PyTorch LLM (non-streaming)
                model_output = self.model.tts(**model_input, speed=speed)
                
                audio_tensor = model_output['tts_speech']
                speech_len = audio_tensor.shape[1] / self.sample_rate
                elapsed = time.time() - start_time
                rtf = elapsed / speech_len if speech_len > 0 else 0
                logging.info(f'Generated speech len={speech_len:.3f}s, rtf={rtf:.3f}')
                yield self._tensor_to_pcm_bytes(audio_tensor)