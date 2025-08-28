# -*- coding: utf-8 -*-
"""
Inferencia ONNX → TensorRT (Jetson Nano / JP 4.6) usando TensorRT + PyCUDA.
- Construye (o carga) un engine .plan desde un .onnx
- Gestiona I/O con PyCUDA (memoria GPU, H2D/D2H)
- Preprocesa imágenes a NCHW 224x224 normalizado
- Devuelve top-K predicciones
"""

import os
import json
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda  # ¡sin pycuda.autoinit!

# ----------------------------
# Utilidad: contexto CUDA por hilo
# ----------------------------
class CudaCtx:
    """
    Crea un contexto CUDA y permite push/pop por hilo.
    Úsalo siempre que llames a CUDA (mem_alloc, memcpy, execute_v2, etc.)
    """
    def __init__(self, device_ordinal=0):
        cuda.init()
        self.dev = cuda.Device(device_ordinal)
        self.ctx = self.dev.make_context()

    def push(self):
        # Hace este contexto “current” en el hilo que llama
        self.ctx.push()

    def pop(self):
        # Quita el contexto del hilo que llama
        self.ctx.pop()

    def detach(self):
        # Destruye el contexto
        self.ctx.detach()


# ----------------------------
# Preprocesado (sin PyTorch)
# ----------------------------
def preprocess_pil(img_pil, size=(224, 224)):
    """RGB PIL -> NCHW float32 normalizado (ImageNet) con batch=1"""
    img = img_pil.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0..1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))
    nchw = np.expand_dims(chw, axis=0).copy()  # asegúrate de contigüidad
    return nchw  # [1,3,H,W] float32


# ---------------------------------------
# Runner TensorRT con **PyCUDA** (no cudart)
# ---------------------------------------
class TrtRunnerPyCUDA(object):
    """
    Ejecuta ONNX/PLAN con TensorRT usando PyCUDA para I/O.
    - Acepta .onnx (construye engine) y .plan (deserializa)
    - Permite fijar perfil estático con batch/h/w (o via ENV TRT_*)
    - Reserva buffers en GPU y ejecuta con execute_v2
    """

    def __init__(self, model_path, fp16=True, save_engine=False,
                 batch=None, h=None, w=None, workspace_mb=None, device_ordinal=0):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None

        self.input_binding = None
        self.output_binding = None
        self.input_shape = None   # (B,C,H,W)
        self.output_shape = None  # (B,K) o lo que emita el modelo
        self.input_dtype = np.float32
        self.output_dtype = np.float32

        self.d_in = None
        self.d_out = None

        self.fp16 = bool(fp16)
        self.save_engine = bool(save_engine)
        self.model_path = model_path
        self.plan_path = (
            model_path if model_path.lower().endswith(".plan")
            else os.path.splitext(model_path)[0] + ".plan"
        )

        # Parámetros de perfil: CLI > ENV > por defecto
        self._B = int(batch if batch is not None else os.getenv("TRT_BATCH", "1"))
        self._H = int(h     if h     is not None else os.getenv("TRT_H",     "224"))
        self._W = int(w     if w     is not None else os.getenv("TRT_W",     "224"))
        self._WS_MB = int(workspace_mb if workspace_mb is not None else os.getenv("TRT_WORKSPACE_MB", "1024"))

        # Crea contexto CUDA (pero NO lo dejamos “puesto” permanente)
        self._cuda = CudaCtx(device_ordinal=device_ordinal)

        # Construye o carga engine
        if model_path.lower().endswith(".plan"):
            self._load_engine_from_plan(self.plan_path)
        elif model_path.lower().endswith(".onnx"):
            self._build_engine_from_onnx(model_path)
        else:
            raise RuntimeError("Ruta no soportada. Usa .onnx o .plan")

        # Prepara bindings y buffers
        self._prepare_io()


    # --- construcción / carga ---

    def _load_engine_from_plan(self, plan_path):
        # Necesita contexto “puesto” en este hilo
        self._cuda.push()
        try:
            runtime = trt.Runtime(self.logger)
            with open(plan_path, "rb") as f:
                engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise RuntimeError("No se pudo deserializar el engine .plan")
            self.context = self.engine.create_execution_context()
        finally:
            self._cuda.pop()

    def _build_engine_from_onnx(self, onnx_path):
        self._cuda.push()
        try:
            B, H, W = self._B, self._H, self._W

            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            builder = trt.Builder(self.logger)
            network = builder.create_network(flag)
            parser = trt.OnnxParser(network, self.logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    msgs = "\n".join(parser.get_error(i).desc() for i in range(parser.num_errors))
                    raise RuntimeError("Fallo al parsear ONNX:\n{}".format(msgs))

            config = builder.create_builder_config()

            # workspace - Optimizado para Jetson Nano
            ws_mb = self._WS_MB
            print(f"[TensorRT] Configurando workspace: {ws_mb}MB")
            
            try:
                # TRT >= 8.6
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_mb * (1 << 20))
            except Exception:
                # TRT 7/8.0 fallback
                config.max_workspace_size = ws_mb * (1 << 20)
            
            # Optimizaciones adicionales para Jetson
            try:
                # Reducir uso de memoria durante la construccion
                config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
                print("[TensorRT] Flags de optimizacion aplicados para Jetson")
            except Exception as e:
                print(f"[TensorRT] Algunas optimizaciones no disponibles: {str(e).encode('ascii', 'replace').decode('ascii')}")
                
            # Optimizaciones adicionales si están disponibles 
            if ws_mb <= 256:
                try:
                    # Intentar optimizaciones agresivas para workspaces pequeños
                    if hasattr(trt.BuilderFlag, 'PREFER_PRECISION_CONSTRAINTS'):
                        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                        print("[TensorRT] Optimizacion de precision aplicada")
                except Exception:
                    pass  # Silencioso si no está disponible

            if self.fp16 and builder.platform_has_fast_fp16:
                try:
                    config.set_flag(trt.BuilderFlag.FP16)
                except Exception:
                    pass

            # Perfil si la entrada es dinámica
            inp = network.get_input(0)
            shape = list(inp.shape)
            if any(dim == -1 for dim in shape):
                profile = builder.create_optimization_profile()
                lo = (B, 3, H, W)
                op = (B, 3, H, W)
                hi = (B, 3, H, W)
                profile.set_shape(inp.name, lo, op, hi)
                config.add_optimization_profile(profile)

            self.engine = builder.build_engine(network, config)
            if self.engine is None:
                raise RuntimeError("No se pudo construir el engine de TensorRT")
            self.context = self.engine.create_execution_context()

            if self.save_engine:
                with open(self.plan_path, "wb") as f:
                    f.write(self.engine.serialize())
        finally:
            self._cuda.pop()

    # --- I/O ---

    def _prepare_io(self):
        self._cuda.push()
        try:
            # Identifica bindings y shapes/dtypes
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))

                bshape = self.engine.get_binding_shape(i)
                if -1 in bshape:
                    # Fija por defecto a [B,3,H,W]
                    ok = self.context.set_binding_shape(i, (self._B, 3, self._H, self._W))
                    if not ok:
                        raise RuntimeError("No se pudo fijar shape dinámica en '{}'".format(name))
                    shape = tuple(self.context.get_binding_shape(i))
                else:
                    shape = tuple(bshape)

                if is_input:
                    self.input_binding = i
                    self.input_shape = shape
                    self.input_dtype = dtype
                else:
                    self.output_binding = i
                    # Para salidas dinámicas, mejor desde el contexto
                    if -1 in bshape:
                        shape = tuple(self.context.get_binding_shape(i))
                    self.output_shape = shape
                    self.output_dtype = dtype

            # Reserva memoria en device
            in_bytes  = int(np.prod(self.input_shape))  * np.dtype(self.input_dtype).itemsize
            out_bytes = int(np.prod(self.output_shape)) * np.dtype(self.output_dtype).itemsize
            self.d_in  = cuda.mem_alloc(in_bytes)
            self.d_out = cuda.mem_alloc(out_bytes)
        finally:
            self._cuda.pop()

    def infer(self, x_nchw):
        """
        x_nchw: np.ndarray [B,3,H,W], dtype esperado por el engine
        Return: logits como np.ndarray con shape self.output_shape
        """
        if not isinstance(x_nchw, np.ndarray):
            raise TypeError("La entrada debe ser numpy.ndarray")

        # Asegura shape/dtype y contigüidad
        if x_nchw.shape != self.input_shape:
            raise ValueError("Shape de entrada {} != esperado {}".format(x_nchw.shape, self.input_shape))
        if x_nchw.dtype != self.input_dtype:
            x_nchw = x_nchw.astype(self.input_dtype, copy=False)
        x_nchw = np.ascontiguousarray(x_nchw)

        self._cuda.push()
        try:
            # H2D
            cuda.memcpy_htod(self.d_in, x_nchw)

            # Ejecutar
            bindings = [None] * self.engine.num_bindings
            bindings[self.input_binding]  = int(self.d_in)
            bindings[self.output_binding] = int(self.d_out)

            ok = self.context.execute_v2(bindings)
            if not ok:
                raise RuntimeError("execute_v2 falló")

            # D2H
            out = np.empty(self.output_shape, dtype=self.output_dtype)
            cuda.memcpy_dtoh(out, self.d_out)
            return out
        finally:
            self._cuda.pop()

    def cleanup(self):
        """Limpia recursos CUDA explicitamente"""
        try:
            self._cuda.push()
            try:
                if self.d_in is not None:
                    self.d_in.free()
                    self.d_in = None
                if self.d_out is not None:
                    self.d_out.free()
                    self.d_out = None
            finally:
                self._cuda.pop()
        except Exception:
            pass
        try:
            self._cuda.detach()
        except Exception:
            pass

    def __del__(self):
        # Intentar liberar con contexto "puesto"
        self.cleanup()


# ---------------------------------------
# Clasificador simple sobre TrtRunner
# ---------------------------------------
class TrtClassifier(object):
    def __init__(self, model_path, labels,
                 fp16=True, save_engine=False,
                 img_size=None, batch=None, device_ordinal=0):
        """
        model_path: ruta a .onnx (compila .plan) o .plan (deserializa)
        labels: lista de nombres de clase
        fp16/save_engine: flags de compilación del engine TRT
        img_size: si lo das, fuerza H=W=img_size para el perfil estático
        batch: tamaño de batch para el perfil (por defecto 1)
        """
        self.labels = list(labels) if labels is not None else []

        h_arg = int(img_size) if img_size else None
        w_arg = int(img_size) if img_size else None
        b_arg = int(batch) if batch else None

        self.runner = TrtRunnerPyCUDA(
            model_path,
            fp16=fp16,
            save_engine=save_engine,
            batch=b_arg,
            h=h_arg,
            w=w_arg,
            device_ordinal=device_ordinal
        )

        # Descubre forma real del binding de entrada (B,C,H,W)
        B, C, H, W = self.runner.input_shape
        if C != 3:
            raise RuntimeError(f"Se esperaba entrada RGB con 3 canales; el modelo expone C={C}")

        self.size = (W, H)  # PIL usa (W,H)
    
    def cleanup(self):
        """Limpia recursos CUDA"""
        try:
            if hasattr(self, 'runner') and self.runner:
                self.runner.cleanup()
        except Exception as e:
            print(f"[WARN] Error limpiando TrtClassifier: {e}")
    
    def __del__(self):
        """Destructor - cleanup automático"""
        self.cleanup()

    def predict_image(self, image_path, topk=3):
        img = Image.open(image_path)
        x = preprocess_pil(img, size=self.size)          # [1,3,H,W] float32
        logits = self.runner.infer(x)                    # [1,num_classes] float32

        # softmax en numpy
        z = logits[0]
        z = z - np.max(z)
        exp = np.exp(z.astype(np.float64))
        probs = (exp / np.sum(exp)).astype(np.float32)

        k = min(topk, probs.shape[0])
        idx = np.argpartition(-probs, k-1)[:k]
        idx = idx[np.argsort(-probs[idx])]

        return [
            {
                "index": int(i),
                "label": (self.labels[i] if i < len(self.labels) else str(i)),
                "prob": float(probs[i]),
            }
            for i in idx
        ]

    def predict_numpy(self, nchw_float32, topk=3):
        logits = self.runner.infer(nchw_float32)
        z = logits[0]
        z = z - np.max(z)
        exp = np.exp(z.astype(np.float64))
        probs = (exp / np.sum(exp)).astype(np.float32)

        k = min(topk, probs.shape[0])
        idx = np.argpartition(-probs, k-1)[:k]
        idx = idx[np.argsort(-probs[idx])]

        return [
            {
                "index": int(i),
                "label": (self.labels[i] if i < len(self.labels) else str(i)),
                "prob": float(probs[i]),
            }
            for i in idx
        ]
