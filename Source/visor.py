import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from PIL import Image, ImageTk
import os
import cv2
import re
from datetime import datetime

class ImageMatrixWindow:
    def __init__(self, master, path):
        self.master = master
        self.path = path
        self.rows = 6
        self.columns = 8
        self.images_per_page = self.rows * self.columns
        self.current_page = 0

        # Maximizar y bloquear la ventana
        # self.master.state("zoomed")  # Maximiza la ventana
        # self.master.resizable(False, False)  # Bloquea el tamaño

        # Crear frame de imágenes
        self.image_matrix_frame = ttk.Frame(self.master)
        self.image_matrix_frame.pack(side='top', fill='both', expand=True)

        # Calcular tamaño de thumbnails una sola vez
        self.master.update_idletasks()  # Asegurar que la ventana se actualice
        frame_width = self.master.winfo_width()
        frame_height = self.master.winfo_height() - 100  # Restamos el espacio del frame de botones
        self.thumbnail_size = (frame_width // self.columns, frame_height // self.rows)

        # Obtener imágenes
        self.update_image_list()

        # Crear etiquetas de imagen
        self.labels = [[ttk.Label(self.image_matrix_frame) for _ in range(self.columns)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.columns):
                label = self.labels[i][j]
                label.grid(row=i, column=j, padx=2, pady=2, sticky='nsew')
                self.image_matrix_frame.grid_columnconfigure(j, weight=1)
                self.image_matrix_frame.grid_rowconfigure(i, weight=1)
                label.bind("<Button-1>", self.on_image_click)

        # Frame de controles (permanece fijo)
        self.button_frame = ttk.Frame(self.master)
        self.button_frame.pack(side='bottom', fill='x')

        self.prev_button = ttk.Button(self.button_frame, text="<< Anterior", command=self.prev_page)
        self.prev_button.pack(side="left", expand=True, fill='x')

        self.page_label = ttk.Label(self.button_frame, text="")
        self.page_label.pack(side="left", expand=True, fill='x')

        self.next_button = ttk.Button(self.button_frame, text="Siguiente >>", command=self.next_page)
        self.next_button.pack(side="left", expand=True, fill='x')

        self.close_button = ttk.Button(self.button_frame, text="Cerrar", command=self.master.destroy)
        self.close_button.pack(side="right", expand=True, fill='x')

        # Cargar imágenes y actualizar cada 5 segundos
        self.load_images()
        self.master.after(5000, self.refresh_images)

    def extract_datetime_from_filename(self, filename):
        match = re.search(r'CAM(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        if match:
            cam_number = match.group(1)
            date_str = match.group(2)
            date_time = datetime.strptime(date_str, '%Y-%m-%d_%H-%M-%S')
            return cam_number, date_time
        return None, None

    def update_image_list(self):
        """Actualiza la lista de imágenes sin modificar la interfaz."""
        # Buscar en directorio de output_images si existe, sino usar path proporcionado
        output_path = '/workspace/output_images' if os.path.exists('/workspace/output_images') else self.path
        if os.path.exists(output_path):
            self.image_files = [f for f in os.listdir(output_path) if f.endswith('.jpg') and 'NOK_CAM' in f]
            self.actual_path = output_path
        else:
            self.image_files = []
            self.actual_path = self.path
        self.image_files.sort(key=lambda f: self.extract_datetime_from_filename(f)[1])
        self.total_pages = max(1, (len(self.image_files) + self.images_per_page - 1) // self.images_per_page)
        self.current_page = min(self.current_page, self.total_pages - 1)

    def load_images(self):
        """Carga imágenes sin reconstruir la interfaz."""
        start_index = self.current_page * self.images_per_page
        end_index = start_index + self.images_per_page
        images_to_show = self.image_files[start_index:end_index]

        for i in range(self.rows):
            for j in range(self.columns):
                label = self.labels[i][j]
                img_index = i * self.columns + j

                # Limpiar imagen anterior para evitar memory leaks
                if hasattr(label, 'image') and label.image:
                    del label.image
                if hasattr(label, 'image_path'):
                    delattr(label, 'image_path')

                if img_index >= len(images_to_show):
                    label.configure(image='', text='')
                    continue

                try:
                    img_path = os.path.join(self.actual_path, images_to_show[img_index])
                    cam_number, date_time = self.extract_datetime_from_filename(images_to_show[img_index])
                    if not cam_number or not date_time:
                        continue

                    # Usar PIL directamente para evitar problemas con cv2
                    image_pil = Image.open(img_path)
                    image_pil = image_pil.resize(self.thumbnail_size, Image.LANCZOS)
                    
                    # Convertir a RGB si es necesario
                    if image_pil.mode != 'RGB':
                        image_pil = image_pil.convert('RGB')
                    
                    photo = ImageTk.PhotoImage(image_pil)

                    label.image_path = img_path
                    label.configure(image=photo, text='')
                    label.image = photo
                    
                    # Limpiar referencias temporales
                    del image_pil
                    
                except Exception as e:
                    print(f"[ERROR] Error cargando imagen {images_to_show[img_index]}: {e}")
                    label.configure(image='', text='Error')
                    continue

        self.page_label.config(text=f"Página {self.current_page + 1} de {self.total_pages}")

    def refresh_images(self):
        """Refresca las imágenes sin destruir la interfaz."""
        self.update_image_list()
        self.load_images()
        self.master.after(5000, self.refresh_images)

    def next_page(self):
        """Avanza a la siguiente página."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_images()

    def prev_page(self):
        """Retrocede a la página anterior."""
        if self.current_page > 0:
            self.current_page -= 1
            self.load_images()

    def on_image_click(self, event):
        """Muestra la imagen en grande cuando se hace clic."""
        label = event.widget
        if not hasattr(label, "image_path"):
            return

        try:
            img_path = label.image_path
            top = tk.Toplevel(self.master)
            top.title("Imagen en grande")
            top.attributes('-zoomed', True)

            screen_width = top.winfo_screenwidth()
            screen_height = top.winfo_screenheight()

            # Usar PIL con manejo de memoria mejorado
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Calcular tamaño manteniendo aspect ratio
            img_ratio = image.width / image.height
            screen_ratio = screen_width / (screen_height - 100)
            
            if img_ratio > screen_ratio:
                new_width = screen_width
                new_height = int(screen_width / img_ratio)
            else:
                new_height = screen_height - 100
                new_width = int((screen_height - 100) * img_ratio)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            main_frame = ttk.Frame(top)
            main_frame.pack(expand=True, fill='both')

            image_label = ttk.Label(main_frame, image=photo)
            image_label.image = photo  # Mantener referencia
            image_label.pack(expand=True, fill='both')

            def close_window():
                # Limpiar memoria antes de cerrar
                if hasattr(image_label, 'image'):
                    del image_label.image
                top.destroy()

            close_button = ttk.Button(top, text="Cerrar", command=close_window)
            close_button.pack(fill='x', side='bottom')
            
            # Limpiar referencia temporal
            del image
            
        except Exception as e:
            print(f"[ERROR] Error mostrando imagen: {e}")
