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

RUN pip3 install opencv-python-headless ultralytics pycuda

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
