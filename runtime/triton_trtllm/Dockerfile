FROM ghcr.io/astral-sh/uv:latest AS uv-builder

FROM nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3

# UV
COPY --from=uv-builder /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

RUN --mount=type=ssh uv pip install --system --break-system-packages --no-cache-dir \
 git+https://github.com/ToolsAiforia/FastCosyVoice.git
