import matplotlib
matplotlib.use('agg')  # Configurar backend ANTES de importar pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import time
import csv
import io
from PIL import Image
import threading
import multiprocessing

class FancyStatsGraphs:
    def __init__(self, labels=None):
        """
        Inicializa FancyStatsGraphs con las labels proporcionadas.
        
        Args:
            labels (list): Lista de etiquetas/categorías del modelo
        """
        # Labels por defecto si no se proporcionan
        if labels is None or not labels:
            labels = ['arandela', 'ok', 'pobres', 'valvula', 'varios']
            print("[WARN] No se proporcionaron labels, usando labels por defecto")
        
        self.labels = labels
        print(f"[INFO] FancyStatsGraphs inicializado con labels: {self.labels}")
        
        # Configurar colores para cada categoría
        self.categories_colors = self._generate_colors(self.labels)
        
        # Lista para almacenar datos (timestamp, category, probability, all_probs)
        self.data = []
        
        # Imágenes PIL para los gráficos
        self.pie_pil = None
        self.timeline_pil = None
        
        print(f"[INFO] Colores asignados: {self.categories_colors}")
    
    def _generate_colors(self, labels):
        """Genera colores únicos para cada label"""
        # Colores predefinidos para labels comunes
        color_map = {
            'ok': '#00ff00',        # Verde
            'arandela': '#ff9900',  # Naranja
            'pobres': '#ff0000',    # Rojo
            'valvula': '#0066cc',   # Azul
            'varios': '#9900cc',    # Morado
            'nok': '#ff0000',       # Rojo
            'good': '#00ff00',      # Verde
            'bad': '#ff0000',       # Rojo
            'defect': '#ff6600',    # Naranja rojizo
            'normal': '#00cc66'     # Verde claro
        }
        
        colors = {}
        default_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', 
                         '#c2c2f0', '#ffb3e6', '#c4e17f', '#76d7c4', '#f7dc6f']
        
        for i, label in enumerate(labels):
            if label.lower() in color_map:
                colors[label] = color_map[label.lower()]
            else:
                # Asignar color por defecto si no hay mapeo específico
                colors[label] = default_colors[i % len(default_colors)]
        
        return colors

    def receive_inference_result(self, result_category, probability_data):
        """
        Recibe resultado de inferencia y lo almacena para generar gráficos.
        
        Args:
            result_category (str): Categoría principal del resultado
            probability_data (list): Lista de tuplas (category, probability)
        """
        timestamp = datetime.now()
        numeric_probability_to_store = float('nan')  # Default to NaN

        try:
            if isinstance(probability_data, list):
                # Formato esperado: lista de tuplas (category_string, probability_score)
                category_found = False
                
                # Almacenar todas las probabilidades para análisis posterior
                all_probs = {}
                for item_cat, item_score in probability_data:
                    if isinstance(item_score, (float, int, np.floating, np.integer)):
                        all_probs[item_cat] = float(item_score)
                        
                        # Si coincide con la categoría principal, guardar su probabilidad
                        if item_cat == result_category:
                            numeric_probability_to_store = float(item_score)
                            category_found = True
                
                if not category_found and result_category:
                    print(f"[WARN] Categoría '{result_category}' no encontrada en datos: {[cat for cat, _ in probability_data]}")
                
                # También almacenar el resultado completo para análisis
                self.data.append((timestamp, result_category, numeric_probability_to_store, all_probs))
                
            elif isinstance(probability_data, (float, int, np.floating, np.integer)):
                # Formato simple: un solo número
                numeric_probability_to_store = float(probability_data)
                self.data.append((timestamp, result_category, numeric_probability_to_store, {result_category: numeric_probability_to_store}))
                
            elif isinstance(probability_data, np.ndarray):
                if probability_data.size == 1:
                    numeric_probability_to_store = float(probability_data.item())
                    self.data.append((timestamp, result_category, numeric_probability_to_store, {result_category: numeric_probability_to_store}))
                else:
                    print(f"[WARN] Array multielemento no soportado: {probability_data.shape}")
                    self.data.append((timestamp, result_category, float('nan'), {}))
            else:
                print(f"[WARN] Formato no reconocido para probability_data: {type(probability_data)}")
                self.data.append((timestamp, result_category, float('nan'), {}))
            
            # Limitar tamaño de datos (mantener últimos 1000 puntos)
            if len(self.data) > 1000:
                self.data = self.data[-1000:]
            
            # Generar gráficos periódicamente
            if len(self.data) % 10 == 0 or len(self.data) <= 5:
                self.generate_graphs()
                
        except Exception as e:
            print(f"[ERROR] Error procesando resultado de inferencia: {e}")
            import traceback
            traceback.print_exc()

    def generate_pie_chart(self):
        """Genera gráfico de torta con distribución de categorías"""
        try:
            category_counts = {}
            
            # Contar ocurrencias de cada categoría
            for entry in self.data:
                if len(entry) >= 2:  # timestamp, category, ...
                    category = entry[1]
                    if category:  # Ignorar categorías vacías
                        category_counts[category] = category_counts.get(category, 0) + 1
            
            if not category_counts:
                # No hay datos, crear gráfico vacío
                plt.figure(figsize=(3, 2.2), facecolor='black')
                plt.text(0.5, 0.5, "Sin datos", horizontalalignment='center', 
                        verticalalignment='center', color='white', fontsize=12)
                plt.axis('off')
            else:
                # Preparar datos para el gráfico
                labels = []
                sizes = []
                colors = []
                
                for category, count in category_counts.items():
                    labels.append(category.upper())
                    sizes.append(count)
                    colors.append(self.categories_colors.get(category, '#808080'))  # Gris por defecto
                
                # Crear gráfico de torta
                fig, ax = plt.subplots(figsize=(3, 2.2), facecolor='black')
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                                 startangle=90, colors=colors)
                
                # Configurar colores del texto
                for text in texts:
                    text.set_color('white')
                    text.set_fontsize(8)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(7)
                
                ax.set_facecolor('black')
                plt.tight_layout()
            
            # Guardar como imagen PIL
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='black', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            self.pie_pil = Image.open(buffer).copy()
            buffer.close()
            plt.close()
            
        except Exception as e:
            print(f"[ERROR] Error generando gráfico de torta: {e}")
            # Crear imagen de error
            try:
                plt.figure(figsize=(3, 2.2), facecolor='black')
                plt.text(0.5, 0.5, "Error gráfico", horizontalalignment='center', 
                        verticalalignment='center', color='red', fontsize=10)
                plt.axis('off')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', facecolor='black', dpi=100)
                buffer.seek(0)
                self.pie_pil = Image.open(buffer).copy()
                buffer.close()
                plt.close()
            except:
                pass

    def generate_timeline_graph(self):
        """Genera gráfico temporal de probabilidades por categoría"""
        try:
            # Filtrar datos de los últimos 10 minutos
            ten_min_ago = datetime.now() - timedelta(minutes=10)
            
            # Extraer datos recientes con probabilidades válidas
            recent_data = []
            for entry in self.data:
                if len(entry) >= 3:  # timestamp, category, probability, ...
                    ts, cat, prob = entry[0], entry[1], entry[2]
                    if ts >= ten_min_ago and not np.isnan(prob) and cat:
                        recent_data.append((ts, cat, prob))
            
            # Crear figura
            fig, ax = plt.subplots(figsize=(3, 2.2), facecolor='black')
            ax.set_facecolor('black')
            
            if not recent_data:
                # No hay datos recientes
                ax.text(0.5, 0.5, 'Sin datos recientes\n(últimos 10 min)', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, color='white', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Plotear datos por categoría
                plotted_categories = set()
                for category_key in self.labels:
                    # Filtrar datos de esta categoría
                    cat_data = [(ts, prob) for ts, cat, prob in recent_data if cat == category_key]
                    
                    if cat_data:
                        times, probs = zip(*cat_data)
                        color = self.categories_colors.get(category_key, '#808080')
                        ax.scatter(times, probs, label=category_key.upper(), 
                                 color=color, alpha=0.7, s=20)
                        plotted_categories.add(category_key)
                
                # Configurar ejes
                ax.set_ylim(0, 1.05)
                ax.set_ylabel('Probabilidad', color='white', fontsize=8)
                ax.tick_params(colors='white', labelsize=7)
                
                # Ocultar etiquetas X (tiempo) para mayor limpieza
                ax.set_xticks([])
                
                # Añadir leyenda si hay datos
                if plotted_categories:
                    legend = ax.legend(loc='upper right', fontsize=6, framealpha=0.8)
                    legend.get_frame().set_facecolor('black')
                    for text in legend.get_texts():
                        text.set_color('white')
                
                # Grid sutil
                ax.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
            
            # Configurar bordes
            for spine in ax.spines.values():
                spine.set_color('white')
                spine.set_linewidth(0.5)
            
            plt.tight_layout()
            
            # Guardar como imagen PIL
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor='black', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            self.timeline_pil = Image.open(buffer).copy()
            buffer.close()
            plt.close()
            
        except Exception as e:
            print(f"[ERROR] Error generando timeline: {e}")
            # Crear imagen de error
            try:
                fig, ax = plt.subplots(figsize=(3, 2.2), facecolor='black')
                ax.set_facecolor('black')
                ax.text(0.5, 0.5, 'Error timeline', horizontalalignment='center', 
                       verticalalignment='center', transform=ax.transAxes, 
                       color='red', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', facecolor='black', dpi=100)
                buffer.seek(0)
                self.timeline_pil = Image.open(buffer).copy()
                buffer.close()
                plt.close()
            except:
                pass

    def generate_graphs(self):
        """Genera ambos gráficos (torta y timeline)"""
        try:
            self.generate_pie_chart()
            self.generate_timeline_graph()
            # print(f"[DEBUG] Gráficos generados - Datos: {len(self.data)} entradas")
        except Exception as error: 
            print(f"[ERROR] No se pudieron generar gráficos: {error}")
            import traceback
            traceback.print_exc()
    
    def get_stats_summary(self):
        """Retorna resumen estadístico de los datos"""
        if not self.data:
            return "Sin datos disponibles"
        
        try:
            category_counts = {}
            total_entries = len(self.data)
            
            for entry in self.data:
                if len(entry) >= 2:
                    category = entry[1]
                    if category:
                        category_counts[category] = category_counts.get(category, 0) + 1
            
            summary = f"Total: {total_entries} entradas\n"
            for cat, count in sorted(category_counts.items()):
                percentage = (count / total_entries) * 100
                summary += f"{cat}: {count} ({percentage:.1f}%)\n"
            
            return summary.strip()
            
        except Exception as e:
            return f"Error generando estadísticas: {e}"
    
    def clear_data(self):
        """Limpia todos los datos almacenados"""
        self.data = []
        print("[INFO] Datos de gráficos limpiados")