import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import PIL.Image, PIL.ImageTk
import os
import threading
import time
import queue

from gun_detector import GunDetector
from descriptor_manager import DescriptorManager
from utils import load_image

class GunDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gun Detection System")
        self.root.geometry("1000x700")
    
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.detector = GunDetector()
        self.descriptor_manager = DescriptorManager()
        
        self.cap = None
        self.is_webcam_active = False
        
        self.current_image = None
        
        self.frame_count = 0
        self.process_every_n_frames = 10
        self.last_result = None
        
        self.processing_queue = queue.Queue(maxsize=1)
        self.detection_thread = None
        self.is_processing = False
        
        self.fps = 0
        self.last_frame_time = 0
        self.frame_times = []
        
        self.webcam_width = 400  
        self.webcam_height = 300
        self.processing_scale = 1.0 
        
        self.create_method_panel()
        self.create_image_panel()
        self.create_control_panel()
        self.create_performance_panel()
        self.create_status_bar()
        
        self.webcam_thread = None
        self.webcam_running = False
        
        self.detection_active = True
        self.start_detection_thread()
    
    def create_method_panel(self):
        self.method_frame = ttk.LabelFrame(self.main_frame, text="Feature Methods")
        self.method_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nw")
        
        self.method_var = tk.StringVar(value="freak")
        
        available_methods = self.descriptor_manager.get_available_methods()
        methods = ['sift', 'orb', 'freak'] if not available_methods else available_methods
        
        for i, method in enumerate(methods):
            ttk.Radiobutton(
                self.method_frame, 
                text=method.upper(), 
                variable=self.method_var,
                value=method
            ).pack(anchor=tk.W, padx=5, pady=5)
        
        self.load_button = ttk.Button(
            self.method_frame, text="Load Selected Method", 
            command=self.load_selected_method
        )
        self.load_button.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        self.refresh_button = ttk.Button(
            self.method_frame, text="Refresh Methods", 
            command=self.refresh_methods
        )
        self.refresh_button.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    
    def create_image_panel(self):
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Image")
        self.image_frame.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="nsew")
        
        self.image_canvas = tk.Canvas(self.image_frame, width=640, height=480, bg="black")
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_canvas.create_text(
            320, 240, text="No image loaded", fill="white", font=("Arial", 20)
        )
    
    def create_control_panel(self):
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nw")
        
        self.load_image_button = ttk.Button(
            self.control_frame, text="Load Image", 
            command=self.load_image
        )
        self.load_image_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.webcam_button = ttk.Button(
            self.control_frame, text="Start Webcam", 
            command=self.toggle_webcam
        )
        self.webcam_button.pack(fill=tk.X, padx=5, pady=5)
        
        self.detect_button = ttk.Button(
            self.control_frame, text="Detect Gun", 
            command=self.detect_gun
        )
        self.detect_button.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', padx=5, pady=10)
        
        exit_button = ttk.Button(
            self.control_frame, 
            text="Exit", 
            command=self.on_closing
        )
        exit_button.pack(fill=tk.X, padx=5, pady=5)
    
    def create_performance_panel(self):
        self.perf_frame = ttk.LabelFrame(self.main_frame, text="Performance Settings")
        self.perf_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nw")
        
        ttk.Label(self.perf_frame, text="Process Every N Frames:").pack(anchor=tk.W, padx=5, pady=2)
        self.frame_skip_var = tk.IntVar(value=self.process_every_n_frames)
        frame_skip_slider = ttk.Scale(
            self.perf_frame, 
            from_=1, 
            to=10, 
            orient=tk.HORIZONTAL, 
            variable=self.frame_skip_var,
            command=self.update_frame_skip
        )
        frame_skip_slider.pack(fill=tk.X, padx=5, pady=2)
        
        self.frame_skip_label = ttk.Label(self.perf_frame, text=f"Current: {self.process_every_n_frames}")
        self.frame_skip_label.pack(anchor=tk.W, padx=5, pady=2)
        
        self.fps_label = ttk.Label(self.perf_frame, text="FPS: 0.0")
        self.fps_label.pack(anchor=tk.W, padx=5, pady=10)
    
    def update_frame_skip(self, value):
        new_value = int(float(value))
        self.process_every_n_frames = new_value
        self.frame_skip_label.config(text=f"Current: {new_value}")
    
    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def refresh_methods(self):
        self.descriptor_manager.load_available_methods()
        available_methods = self.descriptor_manager.get_available_methods()
        self.status_var.set(f"Found {len(available_methods)} methods: {', '.join(available_methods)}")
    
    def load_selected_method(self):
        method = self.method_var.get()
        
        self.status_var.set(f"Loading {method.upper()} descriptors...")
        self.root.update()
        
        def load_thread():
            descriptors = self.descriptor_manager.load_method(method)
            
            self.detector.descriptors[method] = descriptors
            self.detector.set_method(method)
            
            self.root.after(0, lambda: self.status_var.set(f"Loaded {len(descriptors)} {method.upper()} descriptors"))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
        
        try:
            self.current_image = load_image(file_path)
            self.display_image(self.current_image)
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
    
    def toggle_webcam(self):
        if self.is_webcam_active:
            self.webcam_running = False
            if self.webcam_thread:
                self.webcam_thread.join(timeout=1.0)
            
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            
            self.is_webcam_active = False
            self.webcam_button.config(text="Start Webcam")
            self.status_var.set("Webcam stopped")
        else:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.status_var.set("Error: Could not open webcam")
                    return
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
                
                self.is_webcam_active = True
                self.webcam_button.config(text="Stop Webcam")
                self.status_var.set("Webcam started")
                
                self.frame_count = 0
                self.last_result = None
                
                self.webcam_running = True
                self.webcam_thread = threading.Thread(target=self.update_webcam, daemon=True)
                self.webcam_thread.start()
            except Exception as e:
                self.status_var.set(f"Error starting webcam: {str(e)}")
    
    def start_detection_thread(self):
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.detection_active = True
            self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.detection_thread.start()
    
    def detection_worker(self):
        while self.detection_active:
            try:
                try:
                    frame_data = self.processing_queue.get(timeout=0.1)
                    self.is_processing = True
                except queue.Empty:
                    continue
                
                frame, method = frame_data
                
                h, w = frame.shape[:2]
                small_frame = cv2.resize(frame, (int(w * self.processing_scale), int(h * self.processing_scale)))

                is_gun, _, _ = self.detector.detect(small_frame, method=method, visualize_steps=False)
                
                self.last_result = is_gun
                self.root.after(0, lambda: self.update_result_display(is_gun))
                
                self.processing_queue.task_done()
                self.is_processing = False
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Detection error: {str(e)}"))
                self.is_processing = False
                time.sleep(0.1)
    
    def update_result_display(self, is_gun):
        result_text = "GUN DETECTED" if is_gun else "NO GUN"
        self.status_var.set(f"Detection result: {result_text}")
    
    def update_webcam(self):
        last_update_time = time.time()
        frame_interval = 0.03
        
        while self.webcam_running and self.cap and self.cap.isOpened():
            current_time = time.time()
            elapsed = current_time - last_update_time
            
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
                continue
            
            last_update_time = current_time
            
            ret, frame = self.cap.read()
            if not ret:
                self.status_var.set("Error: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = frame_rgb.copy()
            
            self.frame_count += 1
        
            self.update_fps(current_time)
            
            method = self.method_var.get()
            if (self.frame_count % self.process_every_n_frames == 0 and 
                    self.detector.descriptors[method] and 
                    not self.is_processing and 
                    self.processing_queue.empty()):
                try:
                    self.processing_queue.put_nowait((frame_rgb, method))
                except queue.Full:
                    pass
            
            if self.last_result is not None:
                display_frame = frame_rgb.copy()
                self.add_result_text(display_frame, self.last_result)
                self.display_image(display_frame)
            else:
                self.display_image(frame_rgb)
    
    def add_result_text(self, image, is_gun):
        h, w = image.shape[:2]
        
        overlay_height = 100 
        cv2.rectangle(image, (0, 0), (w, overlay_height), (0, 0, 0), -1)
        
        result_text = "GUN DETECTED" if is_gun else "NO GUN"
        color = (255, 0, 0) if is_gun else (0, 255, 0)
        
        font_scale = 1.4
        font_thickness = 3
        
        text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (overlay_height + text_size[1]) // 2 
        
        cv2.putText(
            image, 
            result_text, 
            (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            color, 
            font_thickness
        )
    
    def update_fps(self, current_time):
        if hasattr(self, 'last_frame_time') and self.last_frame_time > 0:
            elapsed = current_time - self.last_frame_time
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                
                self.frame_times.append(current_fps)
                if len(self.frame_times) > 10:
                    self.frame_times.pop(0)
                
                self.fps = sum(self.frame_times) / len(self.frame_times)
                
                if self.frame_count % 5 == 0:
                    self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        
        self.last_frame_time = current_time
    
    def display_image(self, image):
        if image is None:
            return
        
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 640
        if canvas_height <= 1:
            canvas_height = 480
        
        h, w = image.shape[:2]
        
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > 0 and new_h > 0:
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            pil_image = PIL.Image.fromarray(image_resized)
            
            self.tk_image = PIL.ImageTk.PhotoImage(image=pil_image)
    
            self.image_canvas.delete("all")
            
            x = (canvas_width - new_w) // 2
            y = (canvas_height - new_h) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
    
    def detect_gun(self):
        if self.current_image is None:
            self.status_var.set("No image loaded")
            return
        
        method = self.method_var.get()
        if not self.detector.descriptors[method]:
            self.status_var.set(f"No {method.upper()} descriptors loaded. Please load descriptors first.")
            return
        
        self.status_var.set(f"Detecting with {method.upper()}...")
        self.root.update()
        
        try:
            process_image = self.current_image.copy()
            
            h, w = process_image.shape[:2]
            small_image = cv2.resize(process_image, 
                                   (int(w * self.processing_scale), int(h * self.processing_scale)))
            
            is_gun, _, _ = self.detector.detect(small_image, method=method, visualize_steps=False)

            vis_image = self.current_image.copy()
            
            self.add_result_text(vis_image, is_gun)
            
            self.display_image(vis_image)
            
            result_text = "GUN DETECTED" if is_gun else "NO GUN"
            self.status_var.set(f"Detection result ({method.upper()}): {result_text}")
        except Exception as e:
            self.status_var.set(f"Error detecting gun: {str(e)}")
    
    def on_closing(self):
        self.detection_active = False
        self.webcam_running = False
        
        if self.webcam_thread and self.webcam_thread.is_alive():
            self.webcam_thread.join(timeout=1.0)
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = GunDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.bind("<Key>", lambda event: app.on_closing())
    
    root.mainloop()

if __name__ == "__main__":
    main()


