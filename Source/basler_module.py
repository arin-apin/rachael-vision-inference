# basler_module.py
# -*- coding: utf-8 -*-

import cv2
import time
from PIL import Image, ImageTk
import os
from pathlib import Path
import datetime
import numpy as np
import csv
import pandas as pd
# TensorRT classifier (opcional)
try:
    from inference_classification_pytorch_onnx import TrtClassifier
    HAS_TENSORRT = True
except ImportError as e:
    print(f"[WARN] TensorRT no disponible: {e}")
    TrtClassifier = None
    HAS_TENSORRT = False
from graphs_module_v1 import FancyStatsGraphs

# --- Pylon (Basler) ---
try:
    from pypylon import pylon, genicam
except Exception:
    pylon = None
    genicam = None

# Constantes por defecto - detectar si estamos en Docker
if os.path.exists('/workspace'):
    # Entorno Docker
    DEFAULT_MODEL_PATH = '/workspace/models/model.onnx'
    DEFAULT_CONFIG_FILE = '/workspace/source/xx.pfs'
    DEFAULT_EMULATION_PATH = '/workspace/images'
else:
    print('workspace no creado')

def load_labels_from_config(model_path):
    """Intenta cargar labels desde conversion_config.json junto al modelo"""
    import json
    
    model_path = Path(model_path)
    cfg_path = model_path.with_name("conversion_config.json")
    
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            classes = data.get("classes", None)
            if isinstance(classes, list) and classes:
                return [str(c) for c in classes]
        except Exception as e:
            print(f"[WARN] No se pudieron cargar clases desde {cfg_path}: {e}")
    
    # Fallback: labels.txt
    labels_txt = model_path.with_name("labels.txt")
    if labels_txt.exists():
        try:
            with labels_txt.open("r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        except Exception as e:
            print(f"[WARN] No se pudieron cargar clases desde {labels_txt}: {e}")
    
    # Fallback por defecto
    return ['arandela', 'ok', 'pobres', 'valvula', 'varios']

class CamModule:
    def __init__(self, cam_num=1, model_path=None, use_emulation=None, 
                 emu_dir=None, fp16=True, save_engine=True, topk=5, 
                 max_cams=1, emu_fps=10.0):
        
        # Parámetros básicos
        self.cam_number = cam_num
        self.max_cams = max_cams
        self.model_path = model_path or DEFAULT_MODEL_PATH
        
        # Configuración de emulación
        if use_emulation is None:
            env = os.environ.get("PYLON_CAMEMU", "0").strip().lower()
            use_emulation = env in ("1", "true", "yes", "on")
        self.use_emulation = bool(use_emulation)
        self.emu_dir = emu_dir or DEFAULT_EMULATION_PATH
        self.emu_fps = emu_fps
        
        # Cargar labels
        self.labels = load_labels_from_config(self.model_path)
        print(f"[INFO] Labels cargados: {self.labels}")
        
        # Configurar TensorRT - Optimizado para Jetson Nano
        self.classifier = None
        
        # Configurar workspace mediante variable de entorno para TrtRunnerPyCUDA
        workspace_mb = int(os.environ.get('TRT_WORKSPACE_MB', '256'))
        os.environ['TRT_WORKSPACE_MB'] = str(workspace_mb)  # Asegurar que esté en ENV
        
        self._clf_params = {
            'model_path': self.model_path,
            'labels': self.labels,
            'fp16': fp16,
            'save_engine': save_engine
        }
        
        # Configuración de cámara Pylon
        self.converter = None
        self.cameras = []
        if pylon is not None:
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        # Configuraciones de procesamiento
        self.tolerance = 0.70
        self.save = False
        self.time_interval = 0.1  # Reducido de 0.2 a 0.1 para más FPS
        self.topk = topk
        
        # Contadores
        self.cont_ok = 0
        self.cont_nok = 0
        self.frame_count = 0
        self.total = 0
        
        # Directorio de salida para imágenes - usar ruta absoluta para Docker
        if os.path.exists('/workspace'):
            self.output_directory = '/workspace/output_images'
        else:
            self.output_directory = os.path.join(os.getcwd(), 'output_images')
            
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)
            print(f"[INFO] Creado directorio: {self.output_directory}")
        
        # CSV y estadísticas
        self.csv_directory = "csv_outputs"
        if not os.path.exists(self.csv_directory):
            os.makedirs(self.csv_directory)
        
        self.current_date = datetime.datetime.now().date()
        self.csv_file = self.get_csv_file_path()
        self.probability_history = []
        self.save_interval = 100
        
        # Gráficas (pasar labels extraídas)
        self.fancy_stats_graphs = FancyStatsGraphs(labels=self.labels)
        
        # Control de bucle
        self.running = False
        self.update_id = None
        
        # Widgets de interfaz (se configuran después)
        self.image_labels = None
        self.left_frame = None
        self.counters_label = None
        self.graph_pie = None
        self.graph_timeline = None
        self.category_labels = None
        self.category_levels = None
        self.reset_button = None
        
        print(f"[INFO] CamModule inicializado - Modelo: {self.model_path}")
        print(f"[INFO] Emulacion: {'ON' if self.use_emulation else 'OFF'}")
        print(f"[INFO] Directorio emulacion: {self.emu_dir}")
        if self.use_emulation:
            if not os.path.exists(self.emu_dir):
                print(f"[WARN] Directorio de emulacion no existe: {self.emu_dir}")
            else:
                # Contar imágenes disponibles para emulación
                import glob
                image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                image_files = []
                for pattern in image_patterns:
                    image_files.extend(glob.glob(os.path.join(self.emu_dir, pattern)))
                print(f"[INFO] Imagenes disponibles para emulacion: {len(image_files)}")
                
                # Mostrar algunos nombres de archivos para debug
                if image_files:
                    sample_files = [os.path.basename(f) for f in image_files[:5]]
                    print(f"[INFO] Muestra de imagenes: {sample_files}")
                    
                    # Verificar diversidad (buscar diferentes patrones en nombres)
                    ok_files = [f for f in image_files if 'ok' in os.path.basename(f).lower()]
                    nok_files = [f for f in image_files if any(nok_word in os.path.basename(f).lower() 
                               for nok_word in ['nok', 'valvula', 'arandela', 'varios', 'pobres'])]
                    print(f"[INFO] Imagenes OK: {len(ok_files)}, NOK: {len(nok_files)}")
    
    def _ensure_classifier_in_thread(self):
        """Crear clasificador en el hilo correcto para CUDA"""
        if self.classifier is None and HAS_TENSORRT:
            print("[INFO] Inicializando clasificador TensorRT...")
            workspace_mb = int(os.environ.get('TRT_WORKSPACE_MB', '256'))
            print(f"[INFO] Workspace configurado: {workspace_mb}MB")
            
            # Optimizaciones para Jetson Nano
            try:
                # Forzar liberación de memoria GPU si es posible
                import gc
                gc.collect()
                
                # Crear clasificador con configuración optimizada para Jetson
                self.classifier = TrtClassifier(**self._clf_params)
                print("[INFO] Clasificador TensorRT listo")
                
            except RuntimeError as e:
                if "Device memory is insufficient" in str(e) or "oom error" in str(e):
                    print(f"[ERROR] Memoria GPU insuficiente. Reduciendo workspace a 128MB...")
                    # Intentar con workspace más pequeño
                    os.environ['TRT_WORKSPACE_MB'] = '128'
                    try:
                        self.classifier = TrtClassifier(**self._clf_params)
                        print("[INFO] Clasificador TensorRT listo con workspace reducido")
                    except Exception as e2:
                        print(f"[ERROR] Fallo crítico memoria GPU: {e2}")
                        print("[WARN] Continuando sin TensorRT - modo demo")
                        self.classifier = None
                else:
                    print(f"[ERROR] Error TensorRT: {e}")
                    raise
            except Exception as e:
                print(f"[ERROR] Error inesperado TensorRT: {e}")
                raise
                
        elif not HAS_TENSORRT:
            print("[WARN] TensorRT no disponible - funcionando sin inferencia")
    
    def get_csv_file_path(self):
        """Genera nombre de archivo CSV basado en fecha actual"""
        date_str = self.current_date.strftime("%Y-%m-%d")
        return os.path.join(self.csv_directory, f"stats_{date_str}.csv")
    
    def set_widgets(self, image_labels, left_frame, counters_label, graph_pie, 
                   timeline, category_labels, category_levels, reset_button):
        """Configura widgets de la interfaz gráfica"""
        self.image_labels = image_labels
        self.left_frame = left_frame
        self.counters_label = counters_label
        self.graph_pie = graph_pie
        self.graph_timeline = timeline
        self.category_labels = category_labels
        self.category_levels = category_levels
        self.reset_button = reset_button
        print("[INFO] Widgets configurados")
    
    def start_capture(self):
        """Inicia la captura de cámara"""
        if self.running:
            return
        
        try:
            self._open_cameras()
            self.running = True
            self.frame_count = 0
            self.grab_start_time = time.perf_counter()
            
            # Inicializar clasificador en el hilo principal por ahora
            self._ensure_classifier_in_thread()
            
            # Generar imagen de prueba inicial para verificar interfaz
            self._show_test_image()
            
            # Debug: verificar que tenemos cameras y widgets
            camera_count = len(self.cameras) if isinstance(self.cameras, (list, tuple)) else (self.cameras.GetSize() if self.cameras else 0)
            widget_count = len(self.image_labels) if self.image_labels else 0
            print(f"[DEBUG] Iniciando captura con {camera_count} camaras y {widget_count} widgets")
            
            # Iniciar bucle de actualización
            self.update_images()
            print("[INFO] Captura iniciada")
            
        except Exception as e:
            print(f"[ERROR] Error iniciando captura: {e}")
            self.running = False
    
    def _show_test_image(self):
        """Muestra imagen de prueba para verificar que la interfaz funciona"""
        try:
            if self.image_labels and len(self.image_labels) > 0:
                # Crear imagen de prueba
                test_image = np.zeros((480, 640, 3), dtype=np.uint8)
                test_image[:] = (50, 100, 150)  # Color azul grisáceo
                
                # Agregar texto
                cv2.putText(test_image, "RACHAEL - Iniciando...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(test_image, f"Emulacion: {'ON' if self.use_emulation else 'OFF'}", 
                           (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(test_image, f"Modelo: {os.path.basename(self.model_path)}", 
                           (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # Mostrar en interfaz
                self.update_img_label(self.image_labels[0], test_image)
        except Exception as e:
            print(f"[ERROR] Error mostrando imagen de prueba: {e}")
    
    def stop_capture(self):
        """Detiene la captura"""
        self.running = False
        if self.update_id:
            try:
                self.image_labels[0].after_cancel(self.update_id)
            except:
                pass
            self.update_id = None
        print("[INFO] Captura detenida")
    
    def stop_update_loop(self):
        """Alias para compatibilidad"""
        self.stop_capture()
    
    def _open_cameras(self):
        """Abre las cámaras Basler"""
        if pylon is None:
            raise RuntimeError("pypylon no disponible")
        
        # Configurar emulación
        os.environ["PYLON_CAMEMU"] = "1" if self.use_emulation else "0"
        
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        
        if not devices and self.use_emulation:
            time.sleep(0.2)
            devices = tlFactory.EnumerateDevices()
        
        device_count = len(devices) if isinstance(devices, (tuple, list)) else devices.GetSize()
        if device_count < self.cam_number:
            if self.use_emulation:
                print(f'[WARN] Solo {device_count} camaras emuladas disponibles')
            else:
                print(f'export PYLON_CAMEMU={self.cam_number}')
                raise RuntimeError(f"{device_count} camaras presentes, se requieren {self.cam_number}")
        
        # Configurar cámaras
        if self.use_emulation:
            self.cameras = pylon.InstantCameraArray(self.cam_number)
            for i, cam in enumerate(self.cameras):
                device_info = devices[i] if isinstance(devices, (tuple, list)) else devices[i]
                cam.Attach(tlFactory.CreateDevice(device_info))
                cam.Open()
                serial = cam.GetDeviceInfo().GetSerialNumber()
                print(f"[INFO] Usando camara emulada: {serial}")
                
                try:
                    cam.AcquisitionFrameRateEnable.SetValue(True)
                    # Aumentar FPS para emulación más rápida
                    actual_fps = min(30.0, self.emu_fps * 2)  # Doblar FPS pero máximo 30
                    cam.AcquisitionFrameRateAbs.SetValue(actual_fps)
                    cam.TestImageSelector = "Off"
                    cam.ImageFileMode = "On"
                    print(f"[INFO] Configurando emulacion con directorio: {self.emu_dir}")
                    cam.ImageFilename = str(self.emu_dir)
                    cam.Width = 1600
                    cam.Height = 1200
                    print(f"[INFO] Configuracion emulacion aplicada - {actual_fps}fps")
                except Exception as e:
                    print(f"[WARN] Error configurando emulacion: {e}")
        else:
            self.cameras = pylon.InstantCameraArray(min(device_count, self.cam_number))
            for i, cam in enumerate(self.cameras):
                device_info = devices[i] if isinstance(devices, (tuple, list)) else devices[i]
                cam.Attach(tlFactory.CreateDevice(device_info))
                cam.Open()
                print(f"[INFO] Usando camara fisica: {cam.GetDeviceInfo().GetModelName()}")
                
                # Cargar configuracion si existe
                config_file = DEFAULT_CONFIG_FILE
                if os.path.exists(config_file):
                    try:
                        pylon.FeaturePersistence.Load(config_file, cam.GetNodeMap(), True)
                        print(f"[INFO] Configuracion cargada: {config_file}")
                    except Exception as e:
                        print(f"[WARN] Error cargando configuracion: {e}")
        
        # Iniciar grabbing
        for i, cam in enumerate(self.cameras):
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
            time.sleep(0.2)
        
        camera_count = len(self.cameras) if isinstance(self.cameras, (list, tuple)) else self.cameras.GetSize()
        print(f"[INFO] {camera_count} camaras iniciadas")
    
    def release_camera(self):
        """Cierra las cámaras y limpia recursos"""
        # Parar captura primero
        self.running = False
        
        # Cerrar cámaras
        for cam in self.cameras:
            try:
                if cam.IsGrabbing():
                    cam.StopGrabbing()
                if cam.IsOpen():
                    cam.Close()
            except:
                pass
        self.cameras = []
        
        # Limpiar TensorRT/CUDA
        if hasattr(self, 'classifier') and self.classifier:
            try:
                self.classifier.cleanup()
                self.classifier = None
            except Exception as e:
                print(f"[WARN] Error limpiando clasificador: {e}")
        
        print("[INFO] Camaras y recursos liberados")
    
    def update_images(self):
        """Bucle principal de actualización de imágenes"""
        if not self.running:
            return
        
        camera_count = len(self.cameras) if isinstance(self.cameras, (list, tuple)) else (self.cameras.GetSize() if self.cameras else 0)
        if not self.cameras or camera_count == 0:
            return
        
        try:
            for i, cam in enumerate(self.cameras):
                if cam.IsGrabbing():
                    grabResult = cam.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
                    if grabResult.GrabSucceeded():
                        # Convertir imagen
                        image = self.converter.Convert(grabResult)
                        frame = image.GetArray()
                        
                        # Debug cada 50 frames para no saturar log
                        if self.frame_count % 50 == 0:
                            print(f"[DEBUG] Frame {self.frame_count}: {frame.shape}, min={frame.min()}, max={frame.max()}")
                        
                        frame = cv2.resize(frame, (640, 480))
                        
                        # Procesar con inferencia
                        frame = self.process(frame, i)
                        
                        # Redimensionar para interfaz
                        if hasattr(self.left_frame, 'winfo_width') and self.left_frame:
                            try:
                                frame_width = self.left_frame.winfo_width()
                                frame_height = self.left_frame.winfo_height() - 20 * self.cam_number
                                if frame_width > 100:
                                    frame = cv2.resize(frame, (frame_width, frame_height // self.cam_number))
                            except Exception as resize_error:
                                print(f"[WARN] Error redimensionando para interfaz: {resize_error}")
                        
                        # Actualizar imagen en interfaz
                        if self.image_labels and len(self.image_labels) > i:
                            self.update_img_label(self.image_labels[i], frame)
                            if self.frame_count % 50 == 0:
                                print(f"[DEBUG] Imagen actualizada en GUI - frame {self.frame_count}")
                        
                        # Actualizar graficas cada 10 frames
                        if self.frame_count % 10 == 0:
                            self._update_graphs()
                        
                        grabResult.Release()
                    else:
                        print(f"[WARN] Grab fallido en camara {i}")
                else:
                    print(f"[WARN] Camara {i} no esta grabbing")
                        
        except genicam.GenericException as e:
            print(f"[WARN] Error en captura: {e}")
        except Exception as e:
            print(f"[WARN] Error general: {e}")
        
        # Programar siguiente actualización
        if self.running and self.image_labels and len(self.image_labels) > 0:
            try:
                self.update_id = self.image_labels[0].after(int(self.time_interval * 1000), self.update_images)
                # Debug cada 100 frames
                if self.frame_count % 100 == 0:
                    print(f"[DEBUG] Bucle de actualizacion continuando - frame {self.frame_count}")
            except Exception as e:
                print(f"[ERROR] Error programando actualizacion: {e}")
                # Fallback: usar threading para continuar
                import threading
                threading.Timer(self.time_interval, self.update_images).start()
        elif not self.running:
            print("[DEBUG] Bucle detenido - running=False")
        elif not self.image_labels:
            print("[DEBUG] Bucle detenido - no image_labels")
        else:
            print("[DEBUG] Bucle detenido - image_labels vacio")
    
    def process(self, frame, cam_idx):
        """Procesa un frame con inferencia"""
        # Pre-procesamiento básico
        frame[:, :25] = 0
        frame[:, 575:] = 0
        
        # Inferencia - solo cada N frames si es muy lenta para mejorar FPS
        skip_inference = (self.frame_count % 3 != 0)  # Solo inferencia cada 3 frames
        
        try:
            if self.classifier is None:
                self._ensure_classifier_in_thread()
            
            if HAS_TENSORRT and self.classifier and not skip_inference:
                # Debug cada 100 frames para verificar inferencia
                if self.frame_count % 100 == 0:
                    print(f"[DEBUG] Ejecutando inferencia TensorRT - frame {self.frame_count}")
                
                try:
                    # Medir tiempo de inferencia
                    inference_start = time.perf_counter()
                    
                    # Preparar imagen para TensorRT
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    from inference_classification_pytorch_onnx import preprocess_pil
                    x_nchw = preprocess_pil(img_pil, size=self.classifier.size)
                    
                    # Ejecutar inferencia
                    predictions = self.classifier.predict_numpy(x_nchw, topk=self.topk)
                    
                    inference_time = (time.perf_counter() - inference_start) * 1000  # en ms
                    
                    if predictions:
                        top_pred = predictions[0]
                        label = top_pred['label']
                        score = top_pred['prob']
                        
                        # Debug resultado cada 100 frames con tiempo
                        if self.frame_count % 100 == 0:
                            print(f"[DEBUG] Inferencia: {label} ({score:.3f}) - {inference_time:.1f}ms")
                        
                        # Convertir formato para compatibilidad
                        results = [(pred['label'], pred['prob']) for pred in predictions]
                        
                        # Guardar último resultado para frames sin inferencia
                        self._last_inference_result = (results, label)
                        
                        # Determinar si es NOK
                        is_nok = (label != "ok" or score < self.tolerance)
                        
                        if is_nok:
                            self.cont_nok += 1
                            color = (0, 0, 255)  # Rojo
                            
                            # Siempre guardar imagen NOK para el visor
                            self._save_nok_image(frame, cam_idx)
                            
                            # Guardar imagen NOK adicional si está habilitado
                            if self.save:
                                pass  # La función _save_nok_image ya guarda la imagen
                        else:
                            self.cont_ok += 1
                            color = (0, 255, 255)  # Amarillo
                    else:
                        # No predictions fallback
                        results = [('sin_prediccion', 0.0)]
                        label = 'sin_prediccion'
                        self.cont_nok += 1
                        color = (0, 0, 255)
                        if self.frame_count % 100 == 0:
                            print("[DEBUG] No se obtuvieron predicciones")
                            
                except Exception as inference_error:
                    # Error en inferencia - usar fallback
                    if self.frame_count % 100 == 0:
                        print(f"[ERROR] Error en inferencia TensorRT: {inference_error}")
                    results = [('error_inferencia', 0.0)]
                    label = 'error_inferencia'
                    self.cont_nok += 1
                    color = (255, 0, 255)  # Magenta para errores
                    
            elif skip_inference and HAS_TENSORRT and self.classifier:
                # Usar último resultado conocido para mantener velocidad
                if hasattr(self, '_last_inference_result'):
                    results, label = self._last_inference_result
                else:
                    results = [('procesando', 0.5)]
                    label = 'procesando'
                color = (128, 128, 128)  # Gris para frames sin inferencia
                
            else:
                # Fallback sin TensorRT - solo mostrar frame
                results = [('modo_demo', 1.0), ('sin_tensorrt', 0.8)]
                label = 'modo_demo'
                self.cont_ok += 1
                color = (0, 255, 0)  # Verde
                
                if self.frame_count % 100 == 0:
                    classifier_status = "disponible" if HAS_TENSORRT else "no disponible"
                    classifier_init = "inicializado" if self.classifier else "no inicializado"
                    print(f"[DEBUG] Modo demo - TensorRT {classifier_status}, clasificador {classifier_init}")
                
            # Actualizar estadísticas (siempre, independiente de TensorRT)
            if self.fancy_stats_graphs:
                self.fancy_stats_graphs.receive_inference_result(label, results)
            
            # Actualizar contadores en interfaz
            if self.counters_label:
                total = self.cont_nok + self.cont_ok
                total_text = f'TOTAL: {total}   NOK: {self.cont_nok}   OK: {self.cont_ok}'
                self.counters_label.configure(text=total_text)
                
                # Debug contadores cada 100 frames
                if self.frame_count % 100 == 0:
                    print(f"[DEBUG] Contadores: Total={total}, NOK={self.cont_nok}, OK={self.cont_ok}")
            
            # Actualizar barras de progreso
            if self.category_labels and self.category_levels:
                for n, (pred_label, prob) in enumerate(results[:len(self.category_labels)]):
                    prob_percent = prob * 100
                    if n < len(self.category_labels):
                        self.category_labels[n].configure(text=pred_label)
                    if n < len(self.category_levels):
                        self.category_levels[n].configure(value=prob_percent)
                
                # Agregar a historial para CSV
                self.probability_history.append([datetime.datetime.now(), results])
                
                # Guardar CSV periódicamente
                if len(self.probability_history) >= self.save_interval:
                    self.save_to_csv()
                
                # Verificar cambio de día
                self.check_new_csv_file()
                
        except Exception as e:
            print(f"[ERROR] Error en inferencia: {e}")
        
        self.frame_count += 1
        return frame
    
    def _save_nok_image(self, frame, cam_idx):
        """Guarda imagen NOK"""
        try:
            # Limpiar imágenes viejas
            imagenes = sorted(Path(self.output_directory).glob("NOK_CAM*.jpg"), 
                            key=os.path.getmtime, reverse=True)
            if len(imagenes) > 200:
                for imagen in imagenes[200:]:
                    os.remove(str(imagen))
                    print(f'[INFO] Eliminada imagen antigua: {imagen}')
            
            # Guardar nueva imagen
            current_datetime = datetime.datetime.now()
            date_str = current_datetime.strftime("%Y-%m-%d")
            time_str = current_datetime.strftime("%H-%M-%S")
            image_filename = os.path.join(self.output_directory, 
                                        f'NOK_CAM{cam_idx}_{date_str}_{time_str}.jpg')
            cv2.imwrite(image_filename, frame)
            print(f"[INFO] Imagen NOK guardada: {image_filename}")
            
        except Exception as e:
            print(f"[ERROR] Error guardando imagen NOK: {e}")
    
    def _update_graphs(self):
        """Actualiza gráficas en la interfaz"""
        try:
            if not self.fancy_stats_graphs:
                return
                
            # Gráfica circular
            if (self.graph_pie and hasattr(self.fancy_stats_graphs, 'pie_pil') 
                and self.fancy_stats_graphs.pie_pil):
                if isinstance(self.fancy_stats_graphs.pie_pil, Image.Image):
                    image_pie_tk = ImageTk.PhotoImage(image=self.fancy_stats_graphs.pie_pil)
                    self.graph_pie.configure(image=image_pie_tk)
                    self.graph_pie.image = image_pie_tk
            
            # Gráfica timeline  
            if (self.graph_timeline and hasattr(self.fancy_stats_graphs, 'timeline_pil')
                and self.fancy_stats_graphs.timeline_pil):
                if isinstance(self.fancy_stats_graphs.timeline_pil, Image.Image):
                    image_timeline_tk = ImageTk.PhotoImage(image=self.fancy_stats_graphs.timeline_pil)
                    self.graph_timeline.configure(image=image_timeline_tk)
                    self.graph_timeline.image = image_timeline_tk
                    
        except Exception as e:
            print(f"[WARN] Error actualizando graficas: {e}")
    
    def update_img_label(self, img_label, img_bgr):
        """Actualiza label de imagen en Tkinter"""
        try:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            img_label.configure(image=image_tk)
            img_label.image = image_tk  # Mantener referencia
        except Exception as e:
            print(f"[ERROR] Error actualizando label imagen: {e}")
    
    def save_to_csv(self):
        """Guarda resultados en CSV"""
        try:
            if not self.probability_history:
                return
                
            data = []
            categories = set()
            
            for entry in self.probability_history:
                timestamp, values = entry
                row = {'datetime': timestamp}
                for category, probability in values:
                    row[category] = probability
                    categories.add(category)
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Append si el archivo existe
            if os.path.exists(self.csv_file):
                df_existing = pd.read_csv(self.csv_file, parse_dates=['datetime'])
                df = pd.concat([df_existing, df], ignore_index=True)
            
            # Asegurar columnas
            for category in categories:
                if category not in df.columns:
                    df[category] = 0.0
            
            df.to_csv(self.csv_file, index=False)
            print(f"[INFO] CSV actualizado: {self.csv_file}")
            self.probability_history = []
            
        except Exception as e:
            print(f"[ERROR] Error guardando CSV: {e}")
    
    def check_new_csv_file(self):
        """Verifica si cambió el día y crea nuevo CSV"""
        current_date = datetime.datetime.now().date()
        if current_date != self.current_date:
            self.current_date = current_date
            self.csv_file = self.get_csv_file_path()
            print(f"[INFO] Nuevo dia - CSV: {self.csv_file}")