# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import torch
import torchaudio
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, 'r', encoding='utf8') as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr, min_sr=16000):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate >= min_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


@torch.no_grad()
def export_flow_decoder_estimator_onnx(
    estimator: torch.nn.Module,
    onnx_path: str,
    device: torch.device,
    opset_version: int = 20,
    optimize: bool = False,
    fp16: bool = False,
    seq_len_opt: int = 100
):
    """
    Export flow decoder estimator to ONNX format.
    
    Based on cosyvoice/bin/export_onnx.py with optimizations.
    Creates ONNX with 6 inputs: x, mask, mu, t, spks, cond
    
    Key optimizations:
    - opset_version=20 (max for traditional export), 21 with dynamo=True
    - Fixed batch_size=2 for CFG (Classifier-Free Guidance)
    - Dynamic seq_len for streaming inference
    - Optional graph optimizations and FP16 conversion
    
    Args:
        estimator: The DiT or ConditionalDecoder model
        onnx_path: Output path for ONNX file
        device: torch device
        opset_version: ONNX opset version (default 20, max for traditional export)
        optimize: Apply ONNX Runtime graph optimizations
        fp16: Convert to FP16 precision
        seq_len_opt: Optimal sequence length for export (default 100)
    """
    if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
        logging.info(f"ONNX already exists: {onnx_path}")
        return
    
    logging.info(f"Exporting flow decoder estimator to ONNX: {onnx_path}")
    
    # Get model dtype and convert to fp32 for ONNX export
    original_dtype = next(estimator.parameters()).dtype
    estimator_fp32 = estimator.float()
    estimator_fp32.eval()
    
    # Get out_channels from estimator
    out_channels = estimator_fp32.out_channels
    # Fixed batch_size=2 for CFG (Classifier-Free Guidance)
    batch_size = 2
    seq_len = seq_len_opt
    
    logging.info(f"Model out_channels: {out_channels}")
    logging.info(f"Export batch_size: {batch_size} (fixed for CFG)")
    logging.info(f"Export seq_len: {seq_len} (dynamic)")
    
    # Create dummy inputs in fp32 (ONNX export requires fp32)
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size,), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    
    # Dynamic axes - only seq_len varies, batch is fixed to 2 for CFG
    dynamic_axes = {
        'x': {2: 'seq_len'},
        'mask': {2: 'seq_len'},
        'mu': {2: 'seq_len'},
        'cond': {2: 'seq_len'},
        'estimator_out': {2: 'seq_len'},
    }
    
    # CRITICAL: For DiT models, the 'streaming' parameter affects attention mask!
    # CosyVoice3 uses streaming=True in inference, so we MUST export with streaming=True
    streaming = True
    logging.info(f"Export with streaming={streaming} (CRITICAL: bakes DiT attention mask pattern)")
    
    # Create a wrapper that calls estimator with streaming=True
    class EstimatorWrapper(torch.nn.Module):
        def __init__(self, est, stream):
            super().__init__()
            self.estimator = est
            self.streaming = stream
        
        def forward(self, x, mask, mu, t, spks, cond):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=self.streaming)
    
    wrapped_estimator = EstimatorWrapper(estimator_fp32, streaming)
    wrapped_estimator.eval()
    
    # Traditional export (opset 20 max)
    actual_opset = min(opset_version, 20)
    logging.info(f"Using torch.onnx.export (opset {actual_opset})...")
    torch.onnx.export(
        wrapped_estimator,
        (x, mask, mu, t, spks, cond),
        onnx_path,
        export_params=True,
        opset_version=actual_opset,
        do_constant_folding=True,
        input_names=['x', 'mask', 'mu', 't', 'spks', 'cond'],
        output_names=['estimator_out'],
        dynamic_axes=dynamic_axes,
    )
    
    logging.info(f"Initial export complete: {onnx_path}")
    
    # Optional: Apply ONNX Runtime optimizations
    if optimize:
        try:
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions
            
            logging.info("Applying ONNX Runtime graph optimizations...")
            
            fusion_options = FusionOptions('bert')
            fusion_options.enable_attention = True
            fusion_options.enable_layer_norm = True
            fusion_options.enable_gelu = True
            fusion_options.enable_gelu_approximation = True
            
            optimized_model = optimizer.optimize_model(
                onnx_path,
                model_type='bert',
                num_heads=16,  # DiT typically uses 16 heads
                hidden_size=1024,  # DiT dim=1024
                optimization_options=fusion_options,
                opt_level=99,
                use_gpu=True,
            )
            
            if fp16:
                logging.info("Converting to FP16...")
                optimized_model.convert_float_to_float16(
                    keep_io_types=True,
                    force_fp16_initializers=True,
                )
            
            optimized_path = onnx_path.replace('.onnx', '.optimized.onnx')
            optimized_model.save_model_to_file(optimized_path)
            logging.info(f"Optimized model saved to: {optimized_path}")
            
        except ImportError:
            logging.warning("onnxruntime-tools not available, skipping optimization")
        except Exception as e:
            logging.error(f"Optimization failed: {e}", exc_info=True)
    elif fp16:
        try:
            import onnx
            from onnxconverter_common import float16
            
            logging.info("Converting to FP16...")
            model = onnx.load(onnx_path)
            model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
            fp16_path = onnx_path.replace('.onnx', '.fp16.onnx')
            onnx.save(model_fp16, fp16_path)
            logging.info(f"FP16 model saved to: {fp16_path}")
        except ImportError:
            logging.warning("onnxconverter-common not available, skipping FP16 conversion")
        except Exception as e:
            logging.error(f"FP16 conversion failed: {e}", exc_info=True)
    
    # Restore original dtype
    if original_dtype == torch.float16:
        estimator.half()
    
    logging.info(f"Successfully exported ONNX to {onnx_path}")


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt
    logging.info(f"Converting onnx to trt (fp16={fp16})...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # Use WARNING level to see errors but not spam; switch to VERBOSE for debugging
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    # FP32 needs more workspace memory for tactics; use 8GB for FP32, 4GB for FP16
    workspace_size = 1 << 33 if not fp16 else 1 << 32  # 8GB or 4GB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        # Precision constraints are needed if we want to force some layers to FP32 for numerical stability.
        # Different TRT versions expose different flags; try the stricter ones first.
        if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    else:
        # FP32 mode: enable TF32 for better performance on Ampere+ GPUs
        if hasattr(trt.BuilderFlag, "TF32"):
            config.set_flag(trt.BuilderFlag.TF32)
            logging.info("TensorRT: TF32 enabled for FP32 build")
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))

    # FP16 + Transformer attention can be numerically unstable in TensorRT (NaNs), especially around Softmax.
    # Force Softmax layers to run in FP32 while keeping the rest in FP16 for speed.
    if fp16:
        try:
            forced = 0
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                # Some TRT versions use enums; compare by name to be safe.
                layer_type_name = str(getattr(layer, "type", ""))
                if "SOFTMAX" in layer_type_name:
                    layer.precision = trt.DataType.FLOAT
                    for oi in range(layer.num_outputs):
                        try:
                            layer.set_output_type(oi, trt.DataType.FLOAT)
                        except Exception:
                            # Not all layers/versions support setting output types; ignore.
                            pass
                    forced += 1
            if forced > 0:
                logging.info(f"TensorRT: forced {forced} Softmax layers to FP32 for stability (FP16 build).")
            else:
                logging.info("TensorRT: no Softmax layers found to force to FP32 (FP16 build).")
        except Exception as e:
            logging.error(f"TensorRT: failed to apply FP32 Softmax forcing: {e}", exc_info=True)
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        # If we forced some layers (e.g. Softmax) to FP32, TRT may produce FP32 output.
        # Keep output in FP16 only if the network output supports it; otherwise leave as FP32.
        try:
            output_tensor.dtype = tensor_dtype
        except Exception:
            output_tensor.dtype = trt.DataType.FLOAT
    config.add_optimization_profile(profile)
    logging.info(f"TensorRT: building engine for {onnx_model}...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        # Gather diagnostic info
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            gpu_mem = torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0
        except Exception:
            gpu_name, gpu_mem = "Unknown", 0
        trt_version = getattr(trt, '__version__', 'unknown')
        
        error_msg = (
            f"TensorRT failed to build the engine (fp16={fp16}).\n"
            f"TensorRT version: {trt_version}, GPU: {gpu_name} ({gpu_mem}GB)\n"
            f"ONNX model: {onnx_model}\n"
            "Possible causes:\n"
            "  - Insufficient GPU memory (try closing other applications)\n"
            "  - ONNX model contains unsupported operations\n"
            "  - TensorRT version incompatibility\n"
            "Check TensorRT logs above for specific error details."
        )
        logging.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Succesfully convert onnx to trt...")


# NOTE do not support bistream inference as only speech token embedding/head is kept
def export_cosyvoice2_vllm(model, model_path, device):
    if os.path.exists(model_path):
        return

    dtype = torch.bfloat16
    # lm_head
    use_bias = True if model.llm_decoder.bias is not None else False
    model.llm.model.lm_head = model.llm_decoder
    # embed_tokens
    embed_tokens = model.llm.model.model.embed_tokens
    model.llm.model.set_input_embeddings(model.speech_embedding)
    model.llm.model.to(device)
    model.llm.model.to(dtype)
    tmp_vocab_size = model.llm.model.config.vocab_size
    tmp_tie_embedding = model.llm.model.config.tie_word_embeddings
    del model.llm.model.generation_config.eos_token_id
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    model.llm.model.config.vocab_size = model.speech_embedding.num_embeddings
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.use_bias = use_bias
    model.llm.model.save_pretrained(model_path)
    if use_bias is True:
        os.system('sed -i s@Qwen2ForCausalLM@CosyVoice2ForCausalLM@g {}/config.json'.format(os.path.abspath(model_path)))
    model.llm.model.config.vocab_size = tmp_vocab_size
    model.llm.model.config.tie_word_embeddings = tmp_tie_embedding
    model.llm.model.set_input_embeddings(embed_tokens)
