import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle
from basler_module import CamModule
from PIL import Image, ImageTk
from tkinter import messagebox
from visor import ImageMatrixWindow
#from stats import PlotlyGraphWindow
import sys
import os

sys.path.insert(0, os.environ.get("SOURCE_DIR", "/workspace/source"))

class RachaelGUI(ttk.Frame):
    def __init__(self, window, window_title):
        super().__init__(window)

        #self.root = root
        self.root = window  # Asegurate de usar el parametro correctamente
        self.root.geometry("1280x800")  # Establece la resolucion

        self.root.title(window_title)
        
        self.left_frame = ttk.Frame(self.root, style="ModernFrame.TFrame")
        self.left_frame.place(relx=0, rely=0, relwidth=3/4, relheight=1)

        self.right_frame = ttk.Frame(self.root, style="ModernFrame.TFrame")
        self.right_frame.place(relx=3/4, rely=0, relwidth=1/4, relheight=1)

        self.canvas = tk.Canvas(self.left_frame)
        self.canvas.grid(row=0, column=0, sticky="news")

        self.image_labels = []
        self.interval_labels = []
        categories_num = 5
        cam_num=1

        for i in range(cam_num):
            # Create a frame for each set of labels
            imgs_frame = ttk.Frame(self.canvas)
            imgs_frame.grid(row=i, column=0,  sticky="ew")
            
            # Expand frame horizontally
            imgs_frame.columnconfigure(0, weight=1)
            
            # Configure the image label
            image_label = ttk.Label(imgs_frame)
            image_label.grid(row=0, column=0,  sticky="ew")
            self.image_labels.append(image_label)
            
            # Configure the interval label
            interval_label = ttk.Label(imgs_frame, text="Camara "+str(i))
            interval_label.grid(row=1, column=0,  sticky="ew")
            self.interval_labels.append(interval_label)
        
        self.counters_label = ttk.Label(self.right_frame, text="0")
        self.counters_label.pack(anchor="w")


        self.password_entry=None
        self.password_dialog=None
        self.password="3030"
        
        # Create a ttk.Progressbar widget
        self.category_labels = []
        self.category_levels = []
        for i in range(categories_num):
            category_label = ttk.Label(self.right_frame, text=":")
            category_label.pack(anchor="w")
            self.category_labels.append(category_label)
            progressbar = ttk.Progressbar(self.right_frame, orient="horizontal", length=300, mode="determinate")
            progressbar.pack(pady=2)
            self.category_levels.append(progressbar)

        self.timeline = ttk.Label(self.right_frame)
        self.timeline.pack(anchor="center", expand=0)
        image = ImageTk.PhotoImage(image=Image.new('RGB', (300, 100)))
        self.timeline.configure(image=image)
        self.timeline.image = image  


        self.graph_pie = ttk.Label(self.right_frame)
        self.graph_pie.pack(anchor="center", expand=0)
        image = ImageTk.PhotoImage(image=Image.new('RGB', (300, 220)))
        self.graph_pie.configure(image=image)
        self.graph_pie.image = image  



        self.visor_button = ttk.Button(self.right_frame, text="Visor",command=self.show_visor_dialog)
        self.visor_button.pack(fill=tk.BOTH, expand=1)

        # self.plots_button = ttk.Button(self.right_frame, text="Stats",command=self.show_plots_dialog)
        # self.plots_button.pack(fill=tk.BOTH, expand=1)
        
        self.config_button = ttk.Button(self.right_frame, text="Config", command=self.show_password_dialog)
        self.config_button.pack(fill=tk.BOTH, expand=1)
                
        self.quit_button = ttk.Button(self.right_frame, text="Quit", command=self.quit_and_cleanup)
        self.quit_button.pack(fill=tk.BOTH, expand=1)

        # Configurar argumentos para CamModule (compatible con TensorRT y emulacion/camara fisica)
        cam_args = {
            'cam_num': cam_num,
            'max_cams': cam_num,
            'model_path': None,  # usar auto-deteccion desde DEFAULT_MODEL_PATH
            'use_emulation': None,  # usar variable de entorno PYLON_CAMEMU
            'emu_dir': None,  # usar DEFAULT_EMULATION_PATH
            'fp16': True,  # usar FP16 en TensorRT
            'save_engine': True,  # guardar engine .plan
            'topk': categories_num,  # numero de predicciones top-K
            'emu_fps': 10.0  # FPS para emulacion
        }
        
        # Crear e inicializar CamModule
        try:
            self.cam_module = CamModule(**cam_args)
            self.cam_module.set_widgets(self.image_labels, self.left_frame, self.counters_label,
                                       self.graph_pie, self.timeline, self.category_labels, 
                                       self.category_levels, self.visor_button)
            print("[INFO] CamModule configurado exitosamente")
            
            # Iniciar captura
            self.cam_module.start_capture()
            print("[INFO] Captura iniciada desde GUI")
            
        except Exception as e:
            print(f"[ERROR] Error inicializando CamModule: {e}")
            # Mostrar error en interfaz
            error_msg = f"Error: {str(e)}"
            self.counters_label.configure(text=error_msg) 

        # self.stats = MatplotlibGraphWindow(master=self, directorio_csv=self.cam_module.csv_directory)

    # def show_plots_dialog(self):
    #     PlotlyGraphWindow(directorio_csv=self.cam_module.csv_directory)
        
    def quit_and_cleanup(self):
        self.cam_module.release_camera()
        self.cam_module.stop_update_loop()
        self.root.quit()
    
    def check_password(self):
        # Access the global password_entry variable
        entered_password = self.password_entry.get()
        if entered_password == self.password:  
            self.show_config_dialog()
        else:
            self.password_entry.delete(0, tk.END)
            messagebox.showerror("Password Incorrect", "Password is incorrect!")

    def on_change_radio(self):
        if str(self.radio_selected.get())=="save":
            self.cam_module.save=True
            self.config_button.configure(bg="brown1", activebackground="brown1", activeforeground="grey99")
        else:
            self.cam_module.save=False
            self.config_button.configure(bg="grey25", activebackground= "grey25", activeforeground="grey99")
    
    def update_interval_slider(self,*args):
        self.cam_module.tolerance=int(self.slider_tolerance.get())/100
        self.slider_label.configure(text="Tolerance level "+str(int(self.cam_module.tolerance*100)))

    def show_visor_dialog(self):
        try:
            # Obtener directorio de imágenes del cam_module
            output_path = getattr(self.cam_module, 'output_directory', 'output_images')
            print(f"[INFO] Abriendo visor con directorio: {output_path}")
            
            # Verificar que existe el directorio
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
                print(f"[INFO] Directorio creado: {output_path}")
            
            # Usar siempre el visor simple y robusto en lugar del complejo
            visor_toplevel = tk.Toplevel(self.root)
            visor_toplevel.title("Visor - Imagenes NOK")
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            visor_toplevel.geometry(f"{screen_width}x{screen_height}")
            
            # Crear visor robusto
            self._create_robust_image_viewer(visor_toplevel, output_path)
            
        except Exception as e:
            print(f"[ERROR] Error abriendo visor: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"No se pudo abrir el visor: {str(e)}")
    
    def _create_simple_image_viewer(self, parent, image_dir):
        """Crea un visor simple de imágenes como fallback"""
        try:
            import glob
            
            # Frame principal
            main_frame = ttk.Frame(parent)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Label de información
            info_label = ttk.Label(main_frame, text=f"Directorio: {image_dir}")
            info_label.pack(pady=5)
            
            # Buscar imágenes
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            image_files = []
            for pattern in image_patterns:
                image_files.extend(glob.glob(os.path.join(image_dir, pattern)))
            
            if image_files:
                # Ordenar por fecha de modificación (más recientes primero)
                image_files.sort(key=os.path.getmtime, reverse=True)
                
                # Mostrar información
                count_label = ttk.Label(main_frame, text=f"Imágenes encontradas: {len(image_files)}")
                count_label.pack(pady=5)
                
                # Lista de archivos
                listbox_frame = ttk.Frame(main_frame)
                listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)
                
                scrollbar = ttk.Scrollbar(listbox_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set)
                for img_file in image_files:
                    basename = os.path.basename(img_file)
                    listbox.insert(tk.END, basename)
                listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.config(command=listbox.yview)
                
            else:
                no_images_label = ttk.Label(main_frame, text="No se encontraron imágenes NOK")
                no_images_label.pack(pady=20)
            
            # Botón cerrar
            close_btn = ttk.Button(main_frame, text="Cerrar", command=parent.destroy)
            close_btn.pack(pady=10)
            
        except Exception as e:
            print(f"[ERROR] Error en visor simple: {e}")
            # Último fallback - solo un label con error
            error_label = ttk.Label(parent, text=f"Error accediendo a imágenes: {str(e)}")
            error_label.pack(pady=20)

    def show_config_dialog(self):
        if self.password_dialog:
            self.password_dialog.destroy()

        self.config_dialog = tk.Toplevel(self.root)
        self.config_dialog.title("Config Dialog")
        self.radio_selected = tk.StringVar()
        if self.cam_module.save:
            self.radio_selected.set("save")
        else:
            self.radio_selected.set("dont")

        radio_label = tk.Label(self.config_dialog, text="Save images to home dir?")
        radio_label.pack(pady=10)
        save_radio = ttk.Radiobutton(self.config_dialog, text="Save", variable=self.radio_selected, value="save", command=self.on_change_radio)
        dont_save_radio = ttk.Radiobutton(self.config_dialog, text="Don't save", variable=self.radio_selected, value="dont", command=self.on_change_radio)
        save_radio.pack(fill=tk.BOTH, expand=0)
        dont_save_radio.pack(fill=tk.BOTH, expand=0)

        self.slider_label = tk.Label(self.config_dialog, text="Tolerance level "+str(int(self.cam_module.tolerance*100)))
        self.slider_label.pack(pady=10)
        self.slider_tolerance = ttk.Scale(self.config_dialog, from_=0, to=99, value=int(self.cam_module.tolerance*100), orient=tk.HORIZONTAL )
        self.slider_tolerance.bind("<Motion>", self.update_interval_slider)
        self.slider_tolerance.pack(fill=tk.BOTH, expand=0)

        exit_button = ttk.Button(self.config_dialog, text="Exit", command=lambda: self.config_dialog.destroy())
        exit_button.pack(pady=10)

        self.center_window(self.config_dialog)  # Center the password dialog
        self.config_dialog.transient(self.root)  # Make the password dialog dependent on the main window
        self.config_dialog.grab_set()  # Prevent interaction with the parent form
        self.root.wait_window(self.config_dialog)  # Wait for the password dialog to be closed

    def center_window(self, window):
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window_width = window.winfo_width()
        window_height = window.winfo_height()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def show_password_dialog(self):
        self.password_dialog = tk.Toplevel(self.root)
        self.password_dialog.title("Password Dialog")

        password_label = tk.Label(self.password_dialog, text="Enter Password:")
        password_label.pack(pady=10)

        self.password_entry = ttk.Entry(self.password_dialog, show="*")
        self.password_entry.pack()

        # Create a frame for the numerical pad
        numerical_pad_frame = ttk.Frame(self.password_dialog)
        numerical_pad_frame.pack()

        # Create buttons for each numerical digit
        for i in range(1, 10):
            button = ttk.Button(numerical_pad_frame, text=str(i), command=lambda digit=i: self.append_digit(digit))
            button.grid(row=(i - 1) // 3, column=(i - 1) % 3, padx=5, pady=5)

        # Create the "0" button
        zero_button = ttk.Button(numerical_pad_frame, text="0", command=lambda digit=0: self.append_digit(digit))
        zero_button.grid(row=3, column=1, padx=5, pady=5)

        submit_button = ttk.Button(self.password_dialog, text="Submit", command=self.check_password)
        submit_button.pack(pady=10)

        self.center_window(self.password_dialog)  # Center the password dialog
        self.password_dialog.transient(self.root)  # Make the password dialog dependent on the main window
        self.password_dialog.grab_set()  # Prevent interaction with the parent form
        self.root.wait_window(self.password_dialog)  # Wait for the password dialog to be closed


    def append_digit(self, digit):
        current_password = self.password_entry.get()
        updated_password = current_password + str(digit)
        self.password_entry.delete(0, tk.END)
        self.password_entry.insert(0, updated_password)
        
if __name__ == "__main__":
    import sys
    
    # Manejar argumentos básicos
    use_emulation = "--emu" in sys.argv or os.environ.get("PYLON_CAMEMU", "0") == "1"
    if use_emulation:
        print("[INFO] Modo emulacion activado")
        os.environ["PYLON_CAMEMU"] = "1"
    
    # Configurar ventana principal
    root = tk.Tk()
    root.title("RACHAEL - Vision System with TensorRT")
    
    # Configurar tema
    try:
        style = ThemedStyle(root)
        style.set_theme("equilux")
    except Exception as e:
        print(f"[WARN] No se pudo cargar tema: {e}")
    
    # Configurar pantalla completa
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.winfo_toplevel().geometry(f"{screen_width}x{screen_height}")
    
    # Información del sistema al inicio (sin caracteres especiales)
    print("="*60)
    print("RACHAEL - Sistema de Vision con TensorRT")
    print("Configurado para Jetson Nano / MIC-710AI")  
    print("Inferencia: TensorRT + PyCUDA")
    print("Camara: Basler (fisica/emulacion)")
    print("="*60)
    
    try:
        # Crear aplicacion
        app = RachaelGUI(root, "RACHAEL")
        
        # Manejar cierre de ventana
        def on_closing():
            print("[INFO] Cerrando aplicacion...")
            try:
                app.quit_and_cleanup()
            except:
                pass
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Iniciar aplicacion
        root.mainloop()
        
    except Exception as e:
        print(f"[ERROR] Error critico en aplicacion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Aplicacion terminada")
