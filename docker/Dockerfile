# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev python3-gi python3-gst-1.0 build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# pyds master targets DS 9.0 + Python 3.12 — no patches needed
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git /opt/ds_pyapps \
    && cd /opt/ds_pyapps/bindings \
    && git submodule update --init --depth 1 -- 3rdparty/pybind11 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_CXX_FLAGS="-I/usr/include/python3.12" \
    && make -j$(nproc) \
    && find . -name "pyds*.so" -exec cp {} /usr/local/lib/python3.12/dist-packages/ \;

# TensorRT Python bindings — stable; install before other pip layers so
# adding/changing this step doesn't cascade onto the slower installs below.
# Installs only the bindings (not the C++ libs, which the deb package already
# provides), then writes a one-line shim so `import tensorrt as trt` works.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --extra-index-url https://pypi.nvidia.com \
        'tensorrt-cu12-bindings==10.14.1.48.post1' \
    && mkdir -p /usr/local/lib/python3.12/dist-packages/tensorrt \
    && printf 'from tensorrt_bindings.tensorrt import *\n' \
       > /usr/local/lib/python3.12/dist-packages/tensorrt/__init__.py

# pycuda in its own layer — compiles against CUDA headers; keep separate from
# ultralytics so a torch/model update doesn't trigger a recompile.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install pycuda

# opencv + ultralytics — large downloads; pip cache mount means re-installs
# after a layer invalidation skip the network entirely.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install opencv-python-headless ultralytics

# Compile yolo26_decode TRT plugin (SM 75 = GTX 1660 Ti).
# The .so is baked into the image so init_models.py can build the decode
# engine on first container start without any manual pre-steps.
COPY plugins/yolo26_decode/ /tmp/yolo26_decode/
RUN cmake -B /tmp/yolo26_decode/build -S /tmp/yolo26_decode \
        -DCMAKE_CUDA_ARCHITECTURES=75 \
    && cmake --build /tmp/yolo26_decode/build -j$(nproc) \
    && cmake --install /tmp/yolo26_decode/build \
    && rm -rf /tmp/yolo26_decode

WORKDIR /workspace
COPY docker/init_models.py /init_models.py
ENTRYPOINT ["python3", "/init_models.py"]
