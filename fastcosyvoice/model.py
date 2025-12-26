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
FastCosyVoice3Model - True Pipeline TTS Model

Architecture for maximum LLM throughput:
- LLM runs in dedicated thread with its own CUDA stream
- Flow+Hift run in separate thread (doesn't block LLM)  
- Main thread only yields audio from queue

This achieves 80-90% of isolated LLM throughput by never blocking LLM generation.
"""

import os
import queue
import threading
import time
from contextlib import nullcontext
from typing import Generator, Dict

import numpy as np
import torch
from torch.nn import functional as F

from cosyvoice.utils.file_utils import logging, convert_onnx_to_trt
from cosyvoice.utils.common import TrtContextWrapper


class FastCosyVoice3Model:
    """
    True pipeline TTS model with non-blocking LLM.
    
    Pipeline architecture:
    [LLM Thread] → token_queue → [Flow+Hift Thread] → audio_queue → [Main Thread: yield]
    
    Key insight: Flow+Hift run in their own thread, so their blocking operations
    (TensorRT sync, Hift CPU f0_predictor) don't affect LLM at all.
    """
    
    def __init__(
        self,
        llm: torch.nn.Module,
        flow: torch.nn.Module,
        hift: torch.nn.Module,
        fp16: bool = False
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        
        # Token hop length must match training static_chunk_size
        self.token_hop_len = 25
        
        # LLM gets dedicated CUDA stream for true parallelism
        if torch.cuda.is_available():
            self.llm_stream = torch.cuda.Stream(self.device)
        else:
            self.llm_stream = None
    
    def load(self, llm_model: str, flow_model: str, hift_model: str, *, load_llm: bool = True):
        """Load model weights from files.

        Args:
            llm_model: Path to PyTorch LLM weights (.pt)
            flow_model: Path to Flow weights (.pt)
            hift_model: Path to HiFT weights (.pt)
            load_llm: If False, skip loading/moving the PyTorch LLM to GPU.
                Useful when using TRT-LLM (LLM runs outside PyTorch), to reduce VRAM usage.
        """
        if load_llm:
            self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
            self.llm.to(self.device)
            if self.fp16:
                self.llm.half()
            self.llm.eval()

        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device)
        if self.fp16:
            self.flow.half()
        self.flow.eval()
        
        hift_state_dict = {
            k.replace('generator.', ''): v 
            for k, v in torch.load(hift_model, map_location=self.device).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
    
    def get_trt_kwargs(self):
        """Get TensorRT optimization profiles for Flow decoder."""
        # NOTE: max_shape must be large enough to handle prompt_feat + generated mel
        # For zero-shot with long reference audio (~30s) + long output (~60s), need ~6000 frames
        # Must match estimator call signature and TensorRT bindings order in
        # `cosyvoice/flow/flow_matching.py::forward_estimator`.
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2,), (2, 80), (2, 80, 4)]
        opt_shape = [(2, 80, 1000), (2, 1, 1000), (2, 80, 1000), (2,), (2, 80), (2, 80, 1000)]
        max_shape = [(2, 80, 6000), (2, 1, 6000), (2, 80, 6000), (2,), (2, 80), (2, 80, 6000)]
        input_names = ["x", "mask", "mu", "t", "spks", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}
    
    def load_trt(self, flow_decoder_estimator_model: str, flow_decoder_onnx_model: str, trt_concurrent: int = 1, fp16: bool = False):
        """Load TensorRT engine for Flow decoder (significant speedup)."""
        assert torch.cuda.is_available(), 'TensorRT only supports GPU!'
        
        # Auto-export ONNX if missing. Some model releases do not ship this file.
        if not os.path.exists(flow_decoder_onnx_model) or os.path.getsize(flow_decoder_onnx_model) == 0:
            from cosyvoice.utils.file_utils import export_flow_decoder_estimator_onnx
            export_flow_decoder_estimator_onnx(
                estimator=self.flow.decoder.estimator,
                onnx_path=flow_decoder_onnx_model,
                device=self.device,
            )

        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            logging.info(f'Converting ONNX to TensorRT: {flow_decoder_onnx_model} -> {flow_decoder_estimator_model}')
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        
        del self.flow.decoder.estimator
        
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        
        assert estimator_engine is not None, f'Failed to load TensorRT engine: {flow_decoder_estimator_model}'
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)
        logging.info(f'TensorRT engine loaded: {flow_decoder_estimator_model}')
    
    def _llm_job(
        self,
        text: torch.Tensor,
        prompt_text: torch.Tensor,
        llm_prompt_speech_token: torch.Tensor,
        llm_embedding: torch.Tensor,
        tokens_list: list,
        llm_end_flag: dict,
        tokens_lock: threading.Lock
    ):
        """
        LLM token generation - runs in dedicated thread with its own CUDA stream.
        Never blocked by Flow/Hift operations.
        """
        llm_start_time = time.time()
        token_count = 0
        
        # Pre-move tensors to device once
        text_gpu = text.to(self.device)
        text_len_gpu = torch.tensor([text.shape[1]], dtype=torch.int32, device=self.device)
        prompt_text_gpu = prompt_text.to(self.device)
        prompt_text_len_gpu = torch.tensor([prompt_text.shape[1]], dtype=torch.int32, device=self.device)
        prompt_speech_token_gpu = llm_prompt_speech_token.to(self.device)
        prompt_speech_token_len_gpu = torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32, device=self.device)
        embedding_gpu = llm_embedding.to(self.device)
        
        try:
            llm_context = torch.cuda.stream(self.llm_stream) if self.llm_stream else nullcontext()
            
            with llm_context, torch.inference_mode(), torch.amp.autocast('cuda', enabled=self.fp16):
                for token in self.llm.inference(
                    text=text_gpu,
                    text_len=text_len_gpu,
                    prompt_text=prompt_text_gpu,
                    prompt_text_len=prompt_text_len_gpu,
                    prompt_speech_token=prompt_speech_token_gpu,
                    prompt_speech_token_len=prompt_speech_token_len_gpu,
                    embedding=embedding_gpu,
                ):
                    with tokens_lock:
                        tokens_list.append(token)
                    token_count += 1

            llm_duration = time.time() - llm_start_time
            tokens_per_sec = token_count / llm_duration if llm_duration > 0 else 0
            logging.info(
                f'[LLM] duration={llm_duration:.3f}s, tokens={token_count}, tokens/s={tokens_per_sec:.2f} (wall-clock, async)'
            )
            
        except Exception as e:
            logging.error(f'[LLM] Error: {e}', exc_info=True)
        finally:
            # Cleanup GPU tensors
            del text_gpu, text_len_gpu, prompt_text_gpu, prompt_text_len_gpu
            del prompt_speech_token_gpu, prompt_speech_token_len_gpu, embedding_gpu
            llm_end_flag['done'] = True
    
    def _flow_hift_job(
        self,
        tokens_list: list,
        tokens_lock: threading.Lock,
        llm_end_flag: dict,
        audio_queue: queue.Queue,
        flow_prompt_token_gpu: torch.Tensor,
        flow_prompt_token_len_gpu: torch.Tensor,
        prompt_feat_gpu: torch.Tensor,
        prompt_feat_len_gpu: torch.Tensor,
        flow_embedding_gpu: torch.Tensor,
        prompt_token_pad: int
    ):
        """
        Flow + Hift processing - runs in dedicated thread.
        Blocking operations here don't affect LLM at all.
        """
        token_offset = 0
        mel_cache = None
        speech_offset = 0
        
        # Timing accumulators
        total_flow_time = 0.0
        total_hift_time = 0.0
        flow_call_count = 0
        hift_call_count = 0
        
        try:
            with torch.inference_mode():
                while True:
                    # Minimal polling
                    time.sleep(0.0005)
                    
                    # Calculate required tokens
                    this_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                    required_tokens = this_hop_len + self.flow.pre_lookahead_len
                    
                    # Check token availability (thread-safe)
                    with tokens_lock:
                        current_count = len(tokens_list)
                        tokens_snapshot = list(tokens_list) if current_count > token_offset else None
                    
                    available = current_count - token_offset
                    
                    if available >= required_tokens and tokens_snapshot is not None:
                        batch_end = token_offset + required_tokens
                        
                        # Flow inference (may block on TensorRT sync - that's OK here!)
                        batch_tokens = torch.tensor(tokens_snapshot[:batch_end], dtype=torch.int32, device=self.device).unsqueeze(0)
                        token_len_gpu = torch.tensor([batch_tokens.shape[1]], dtype=torch.int32, device=self.device)
                        
                        flow_start = time.time()
                        tts_mel, _ = self.flow.inference(
                            token=batch_tokens,
                            token_len=token_len_gpu,
                            prompt_token=flow_prompt_token_gpu,
                            prompt_token_len=flow_prompt_token_len_gpu,
                            prompt_feat=prompt_feat_gpu,
                            prompt_feat_len=prompt_feat_len_gpu,
                            embedding=flow_embedding_gpu,
                            streaming=True,
                            finalize=False
                        )
                        flow_elapsed = time.time() - flow_start
                        total_flow_time += flow_elapsed
                        flow_call_count += 1
                        
                        # Extract new mel and accumulate (original logic - Flow needs full history)
                        mel_new = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
                        mel = mel_new if mel_cache is None else torch.cat([mel_cache, mel_new], dim=2)
                        mel_cache = mel
                        
                        # Hift inference (may block on CPU f0_predictor - that's OK here!)
                        hift_start = time.time()
                        tts_speech, _ = self.hift.inference(
                            speech_feat=mel.float(),
                            finalize=False
                        )
                        hift_elapsed = time.time() - hift_start
                        total_hift_time += hift_elapsed
                        hift_call_count += 1
                        
                        # Extract new audio and send to queue
                        audio = tts_speech[:, speech_offset:]
                        speech_offset += audio.shape[1]
                        if audio.is_cuda:
                            audio_cpu = torch.empty_like(audio, device='cpu', pin_memory=True)
                            audio_cpu.copy_(audio, non_blocking=True)
                            ready = torch.cuda.Event()
                            ready.record(torch.cuda.current_stream())
                            audio_queue.put({'tts_speech': audio_cpu, '_ready_event': ready})
                        else:
                            audio_queue.put({'tts_speech': audio})
                        token_offset += this_hop_len
                    
                    # Check if we're done
                    if llm_end_flag['done'] and available < required_tokens:
                        break
                
                # Final batch
                with tokens_lock:
                    final_tokens = list(tokens_list)
                
                if final_tokens and token_offset < len(final_tokens):
                    # Final Flow
                    final_tokens_gpu = torch.tensor(final_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
                    final_token_len = torch.tensor([final_tokens_gpu.shape[1]], dtype=torch.int32, device=self.device)
                    
                    flow_start = time.time()
                    tts_mel, _ = self.flow.inference(
                        token=final_tokens_gpu,
                        token_len=final_token_len,
                        prompt_token=flow_prompt_token_gpu,
                        prompt_token_len=flow_prompt_token_len_gpu,
                        prompt_feat=prompt_feat_gpu,
                        prompt_feat_len=prompt_feat_len_gpu,
                        embedding=flow_embedding_gpu,
                        streaming=True,
                        finalize=True
                    )
                    flow_elapsed = time.time() - flow_start
                    total_flow_time += flow_elapsed
                    flow_call_count += 1
                    
                    mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
                    if mel_cache is not None:
                        mel = torch.cat([mel_cache, mel], dim=2)
                    
                    # Final Hift
                    hift_start = time.time()
                    tts_speech, _ = self.hift.inference(
                        speech_feat=mel.float(),
                        finalize=True
                    )
                    hift_elapsed = time.time() - hift_start
                    total_hift_time += hift_elapsed
                    hift_call_count += 1
                    
                    audio = tts_speech[:, speech_offset:]
                    if audio.is_cuda:
                        audio_cpu = torch.empty_like(audio, device='cpu', pin_memory=True)
                        audio_cpu.copy_(audio, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(torch.cuda.current_stream())
                        audio_queue.put({'tts_speech': audio_cpu, '_ready_event': ready})
                    else:
                        audio_queue.put({'tts_speech': audio})
        
        except Exception as e:
            logging.error(f'[Flow+Hift] Error: {e}', exc_info=True)
        finally:
            # Log timing statistics
            logging.info(
                f'[Flow] total_time={total_flow_time:.3f}s, calls={flow_call_count}, '
                f'avg_per_call={total_flow_time/flow_call_count if flow_call_count > 0 else 0:.3f}s'
            )
            logging.info(
                f'[HiFT] total_time={total_hift_time:.3f}s, calls={hift_call_count}, '
                f'avg_per_call={total_hift_time/hift_call_count if hift_call_count > 0 else 0:.3f}s'
            )
            # Signal completion
            audio_queue.put(None)
    
    def tts_stream(
        self,
        text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        llm_embedding: torch.Tensor = torch.zeros(0, 192),
        prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        **kwargs
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        True pipeline streaming TTS.
        
        Architecture:
        - LLM Thread: generates tokens (never blocked by Flow/Hift)
        - Flow+Hift Thread: processes tokens into audio
        - Main Thread: yields audio chunks from queue
        
        This achieves ~80-90% of isolated LLM throughput.
        """
        # Shared state
        tokens: list = []
        tokens_lock = threading.Lock()
        llm_end_flag = {'done': False}
        audio_queue: queue.Queue = queue.Queue(maxsize=4)  # Small buffer to limit memory
        
        # Pre-compute padding
        prompt_token_pad = int(
            np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) 
            * self.token_hop_len 
            - flow_prompt_speech_token.shape[1]
        )
        
        # Pre-move tensors to device
        dtype = torch.float16 if self.fp16 else torch.float32
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        
        # Start LLM thread
        llm_thread = threading.Thread(
            target=self._llm_job,
            args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, 
                  tokens, llm_end_flag, tokens_lock),
            daemon=True
        )
        
        # Start Flow+Hift thread
        flow_hift_thread = threading.Thread(
            target=self._flow_hift_job,
            args=(tokens, tokens_lock, llm_end_flag, audio_queue,
                  flow_prompt_token_gpu, flow_prompt_token_len_gpu,
                  prompt_feat_gpu, prompt_feat_len_gpu, flow_embedding_gpu,
                  prompt_token_pad),
            daemon=True
        )
        
        llm_thread.start()
        flow_hift_thread.start()
        
        try:
            # Main thread just yields from queue
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=30.0)
                    if audio_chunk is None:  # Sentinel - processing complete
                        break
                    ready = audio_chunk.pop('_ready_event', None)
                    if ready is not None:
                        ready.synchronize()
                    yield audio_chunk
                except queue.Empty:
                    logging.warning('[Main] Timeout waiting for audio chunk')
                    break
            
            # Wait for threads
            llm_thread.join(timeout=5.0)
            flow_hift_thread.join(timeout=5.0)
            
        finally:
            pass  # GPU tensors will be freed when function exits
    
    def tts_stream_external_llm(
        self,
        tokens_list: list,
        tokens_lock: 'threading.Lock',
        llm_end_flag: dict,
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        **kwargs
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        True pipeline streaming TTS with external LLM (e.g. TRT-LLM).
        
        Similar to tts_stream but LLM runs in an external thread.
        Flow+Hift runs in its own thread and processes tokens as they arrive.
        
        Args:
            tokens_list: Shared list where external LLM appends tokens
            tokens_lock: Lock for thread-safe access to tokens_list
            llm_end_flag: Dict with 'done' key set to True when LLM finishes
            flow_embedding: Speaker embedding for flow
            flow_prompt_speech_token: Prompt speech tokens for flow
            prompt_speech_feat: Prompt speech features
        
        Yields:
            Audio chunks as dictionaries with 'tts_speech' key
        """
        # Audio queue for output
        audio_queue: queue.Queue = queue.Queue(maxsize=4)
        
        # Pre-compute padding
        prompt_token_pad = int(
            np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) 
            * self.token_hop_len 
            - flow_prompt_speech_token.shape[1]
        )
        
        # Pre-move tensors to device
        dtype = torch.float16 if self.fp16 else torch.float32
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        
        # Start Flow+Hift thread
        flow_hift_thread = threading.Thread(
            target=self._flow_hift_job,
            args=(tokens_list, tokens_lock, llm_end_flag, audio_queue,
                  flow_prompt_token_gpu, flow_prompt_token_len_gpu,
                  prompt_feat_gpu, prompt_feat_len_gpu, flow_embedding_gpu,
                  prompt_token_pad),
            daemon=True
        )
        flow_hift_thread.start()
        
        try:
            # Main thread yields audio from queue
            while True:
                try:
                    audio_chunk = audio_queue.get(timeout=30.0)
                    if audio_chunk is None:  # Sentinel - processing complete
                        break
                    ready = audio_chunk.pop('_ready_event', None)
                    if ready is not None:
                        ready.synchronize()
                    yield audio_chunk
                except queue.Empty:
                    logging.warning('[Main] Timeout waiting for audio chunk')
                    break
            
            flow_hift_thread.join(timeout=5.0)
            
        finally:
            pass  # GPU tensors will be freed when function exits
    
    def tts(
        self,
        text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        llm_embedding: torch.Tensor = torch.zeros(0, 192),
        prompt_text: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        speed: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Non-streaming TTS inference.
        
        Generates all speech tokens first, then converts them to audio in one pass.
        This is simpler and can be slightly faster for short texts, but has higher
        latency to first audio compared to streaming.
        
        Args:
            text: Input text tokens
            flow_embedding: Speaker embedding for flow
            llm_embedding: Speaker embedding for LLM
            prompt_text: Prompt text tokens
            llm_prompt_speech_token: Prompt speech tokens for LLM
            flow_prompt_speech_token: Prompt speech tokens for flow
            prompt_speech_feat: Prompt speech features
            speed: Speech speed multiplier (1.0 = normal)
        
        Returns:
            Dict with 'tts_speech' key containing full audio tensor [1, audio_len]
        """
        # Shared state for LLM thread
        tokens: list = []
        tokens_lock = threading.Lock()
        llm_end_flag = {'done': False}
        
        # Start LLM thread
        llm_thread = threading.Thread(
            target=self._llm_job,
            args=(text, prompt_text, llm_prompt_speech_token, llm_embedding,
                  tokens, llm_end_flag, tokens_lock),
            daemon=True
        )
        llm_thread.start()
        
        # Wait for LLM to finish generating all tokens
        llm_thread.join()
        
        # Get all tokens
        with tokens_lock:
            all_tokens = list(tokens)
        
        if not all_tokens:
            logging.warning('[TTS] No tokens generated')
            return {'tts_speech': torch.zeros(1, 0)}
        
        # Convert tokens to audio in one pass
        dtype = torch.float16 if self.fp16 else torch.float32
        tokens_gpu = torch.tensor(all_tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
        token_len_gpu = torch.tensor([tokens_gpu.shape[1]], dtype=torch.int32, device=self.device)
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32, device=self.device)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32, device=self.device)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        
        with torch.inference_mode():
            # Flow inference - non-streaming mode
            flow_start = time.time()
            tts_mel, _ = self.flow.inference(
                token=tokens_gpu,
                token_len=token_len_gpu,
                prompt_token=flow_prompt_token_gpu,
                prompt_token_len=flow_prompt_token_len_gpu,
                prompt_feat=prompt_feat_gpu,
                prompt_feat_len=prompt_feat_len_gpu,
                embedding=flow_embedding_gpu,
                streaming=False,
                finalize=True
            )
            flow_elapsed = time.time() - flow_start
            logging.info(f'[Flow] non-streaming inference: {flow_elapsed:.3f}s')
            
            # Apply speed change if requested
            if speed != 1.0:
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            
            # Hift inference - non-streaming mode
            hift_start = time.time()
            tts_speech, _ = self.hift.inference(
                speech_feat=tts_mel.float(),
                finalize=True
            )
            hift_elapsed = time.time() - hift_start
            logging.info(f'[HiFT] non-streaming inference: {hift_elapsed:.3f}s')
        
        return {'tts_speech': tts_speech.cpu()}
    
    def tts_with_external_tokens(
        self,
        tokens: list,
        flow_embedding: torch.Tensor = torch.zeros(0, 192),
        flow_prompt_speech_token: torch.Tensor = torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor = torch.zeros(1, 0, 80),
        speed: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Non-streaming TTS with pre-generated tokens (for TRT-LLM).
        
        Args:
            tokens: Pre-generated speech tokens
            flow_embedding: Speaker embedding for flow
            flow_prompt_speech_token: Prompt speech tokens for flow
            prompt_speech_feat: Prompt speech features
            speed: Speech speed multiplier (1.0 = normal)
        
        Returns:
            Dict with 'tts_speech' key containing full audio tensor [1, audio_len]
        """
        if not tokens:
            logging.warning('[TTS] No tokens provided')
            return {'tts_speech': torch.zeros(1, 0)}
        
        # Convert tokens to audio in one pass
        dtype = torch.float16 if self.fp16 else torch.float32
        tokens_gpu = torch.tensor(tokens, dtype=torch.int32, device=self.device).unsqueeze(0)
        token_len_gpu = torch.tensor([tokens_gpu.shape[1]], dtype=torch.int32, device=self.device)
        flow_prompt_token_gpu = flow_prompt_speech_token.to(self.device)
        flow_prompt_token_len_gpu = torch.tensor([flow_prompt_speech_token.shape[1]], dtype=torch.int32, device=self.device)
        prompt_feat_gpu = prompt_speech_feat.to(self.device, dtype=dtype)
        prompt_feat_len_gpu = torch.tensor([prompt_speech_feat.shape[1]], dtype=torch.int32, device=self.device)
        flow_embedding_gpu = flow_embedding.to(self.device, dtype=dtype)
        
        with torch.inference_mode():
            # Flow inference - non-streaming mode
            flow_start = time.time()
            tts_mel, _ = self.flow.inference(
                token=tokens_gpu,
                token_len=token_len_gpu,
                prompt_token=flow_prompt_token_gpu,
                prompt_token_len=flow_prompt_token_len_gpu,
                prompt_feat=prompt_feat_gpu,
                prompt_feat_len=prompt_feat_len_gpu,
                embedding=flow_embedding_gpu,
                streaming=False,
                finalize=True
            )
            flow_elapsed = time.time() - flow_start
            logging.info(f'[Flow] non-streaming inference: {flow_elapsed:.3f}s')
            
            # Apply speed change if requested
            if speed != 1.0:
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            
            # Hift inference - non-streaming mode
            hift_start = time.time()
            tts_speech, _ = self.hift.inference(
                speech_feat=tts_mel.float(),
                finalize=True
            )
            hift_elapsed = time.time() - hift_start
            logging.info(f'[HiFT] non-streaming inference: {hift_elapsed:.3f}s')
        
        return {'tts_speech': tts_speech.cpu()}