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

WORKDIR /workspace
