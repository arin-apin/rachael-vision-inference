# Imagen de NVIDIA con PyTorch + TF + TensorRT para L4T R32.6.1 (JetPack 4.6.1)
FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Dependencias del sistema (incluye Tk si usas tkinter)
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git \
    python3-opencv libopencv-dev \
    libglib2.0-0 libgl1-mesa-glx \
    python3-tk \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
   

# Herramientas Python compatibles con Py3.6 en JP4.6
RUN python3 -m pip install --upgrade pip==23.2.1 setuptools==65.7.0 wheel==0.38.4 && \
    python3 -m pip install --no-cache-dir \
      numpy==1.19.5 pillow==9.0.1 ttkthemes==3.2.2 \
      onnx==1.10.2 cuda-python==12.2.0

# --- Instalar pypylon desde pip ---
RUN python3 -m pip install --no-cache-dir pypylon

# Estructura de trabajo y volúmenes (se montarán desde docker-compose)
WORKDIR /workspace
RUN mkdir -p /workspace/models /workspace/images /workspace/source /workspace/output_images 

# Variables de entorno típicas para la app 
ENV SOURCE_DIR=/workspace/source \
    MODEL_PATH=/workspace/models/model.onnx \
    LABELS_PATH=/workspace/models/labels.txt \
    IMAGES_PATH=/workspace/images

CMD ["python3", "/workspace/source/RACHAEL.py"]

