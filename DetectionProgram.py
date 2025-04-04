import cv2
import numpy as np
import customtkinter as ctk
import threading
import os
import tensorflow as tf
from tkinter import messagebox
from pygrabber.dshow_graph import FilterGraph
from PIL import Image, ImageTk, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

class DetectionProgram:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")
        self.root.geometry("1280x720")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_source = None
        self.vid = None
        self.running = False
        self.ai_enabled = False
        self.frame_count = 0
        self.input_size = (128, 128)

        # Load the autoencoder model
        #self.model_path = os.path.join(os.path.dirname(__file__), "models", "decoder.h5")
        #if os.path.exists(self.model_path):
        #    self.model = load_model(self.model_path, compile=False)
        #else:
        #    messagebox.showerror("Error", "Model file not found: decoder.h5")
        #    self.model = None

        self.model_list = self.get_model_list()
        self.model = None
        #if self.model_files:
        self.load_selected_model(self.model_list[0])  # Load the first model by default

        # Load feature extractor (MobileNetV2 without top layers)s
        base_model = MobileNetV2(include_top=False, input_shape=(128, 128, 3))
        self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer("block_16_project").output)

        # UI - Top Frame
        self.top_frame = ctk.CTkFrame(root)
        self.top_frame.pack(fill="x", pady=5)

        self.dropdown_label = ctk.CTkLabel(self.top_frame, text="Select Camera:", font=("Arial", 14))
        self.dropdown_label.pack(side="left", padx=5)

        self.cameras = self.get_camera_list()
        self.camera_dropdown = ctk.CTkComboBox(self.top_frame, values=self.cameras)
        self.camera_dropdown.pack(side="left", padx=5)

        self.start_button = ctk.CTkButton(self.top_frame, text="Start", command=self.start_camera)
        self.start_button.pack(side="left", padx=5)

        self.stop_button = ctk.CTkButton(self.top_frame, text="Stop", command=self.stop_camera, state="disabled")
        self.stop_button.pack(side="left", padx=5)

        self.capture_button = ctk.CTkButton(self.top_frame, text="Take Picture", command=self.take_picture, state="disabled")
        self.capture_button.pack(side="left", padx=5)

        self.ai_toggle = ctk.CTkButton(self.top_frame, text="Enable AI", command=self.toggle_ai)
        self.ai_toggle.pack(side="left", padx=5)

        # Model selection dropdown
        self.model_label = ctk.CTkLabel(self.top_frame, text="Select Model:", font=("Arial", 14))
        self.model_label.pack(side="left", padx=5)

        self.model_list = self.get_model_list()
        self.model_dropdown = ctk.CTkComboBox(self.top_frame, values=self.model_list, command=self.load_selected_model)
        self.model_dropdown.pack(side="left", padx=5)

        # Load default model if available
        if self.model_list:
            default_model = "decoder.h5"
            if default_model in self.model_list:
                self.model_dropdown.set(default_model)
                self.load_selected_model(default_model)
            else:
                self.model_dropdown.set(self.model_list[0])
                self.load_selected_model(self.model_list[0])

        # Camera Feed Display
        self.canvas = ctk.CTkLabel(root, text="")
        self.canvas.pack(pady=10, fill="both", expand=True)

        # Create Heatmap Legend
        self.legend_image = self.create_legend()
        self.legend_photo = ImageTk.PhotoImage(self.legend_image)

        self.legend_label = ctk.CTkLabel(root, image=self.legend_photo, text="")
        self.legend_label.pack(side="right", padx=10, pady=20)

        self.last_ai_frame = None

    def get_camera_list(self):
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return devices if devices else ["No camera found"]
    
    def get_model_list(self):
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if not os.path.exists(models_dir):
            return ["decoder.h5"]
        models = [f for f in os.listdir(models_dir) if f.endswith(".h5")]
        return models if models else ["decoder.h5"]
    
    def load_selected_model(self, selected_model=None):
        if not selected_model:
            selected_model = self.model_dropdown.get()

        model_path = os.path.join(os.path.dirname(__file__), "models", selected_model)
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path, compile=False)
                messagebox.showinfo("Model Loaded", f"Loaded model: {selected_model}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
        else:
            messagebox.showerror("Error", f"Model file not found: {selected_model}")

    def start_camera(self):
        selected_camera = self.camera_dropdown.get()
        if not selected_camera or selected_camera == "No camera found":
            messagebox.showerror("Error", "No camera selected or available.")
            return

        camera_index = self.cameras.index(selected_camera)
        self.video_source = camera_index
        self.vid = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)

        if not self.vid.isOpened():
            messagebox.showerror("Error", "Unable to open camera.")
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.capture_button.configure(state="normal")
        self.running = True

        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_camera(self):
        self.running = False
        if self.vid:
            self.vid.release()
            self.vid = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.capture_button.configure(state="disabled")
        self.canvas.configure(image='')

    def update_frame(self):
        while self.running and self.vid.isOpened():
            ret, frame = self.vid.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.ai_enabled:
                if self.frame_count % 20 == 0:
                    self.last_ai_frame = self.apply_ai_detection(frame)
                if self.last_ai_frame is not None:
                    frame = self.last_ai_frame

            frame = Image.fromarray(frame)
            frame_tk = ImageTk.PhotoImage(frame)
            self.canvas.configure(image=frame_tk)
            self.canvas.image = frame_tk

            self.frame_count += 1

    def apply_ai_detection(self, frame):
        if self.model is None:
            return frame

        input_frame = cv2.resize(frame, self.input_size)
        input_frame = input_frame.astype("float32") / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)

        feature_maps = self.feature_extractor.predict(input_frame)
        feature_maps_resized = tf.image.resize(feature_maps, (7, 7)).numpy()
        padding_depth = 1280 - feature_maps_resized.shape[-1]
        feature_maps_padded = np.pad(feature_maps_resized, ((0, 0), (0, 0), (0, 0), (0, padding_depth)))

        reconstructed = self.model.predict(feature_maps_padded)[0]
        reconstructed = np.clip(reconstructed, 0.0, 1.0)

        # Ensure reconstructed has 3 channels
        if reconstructed.ndim == 2:
            reconstructed = np.stack([reconstructed] * 3, axis=-1)
        elif reconstructed.shape[-1] == 1:
            reconstructed = np.repeat(reconstructed, 3, axis=-1)

        original = cv2.resize(frame, (reconstructed.shape[1], reconstructed.shape[0]))
        original = original.astype("float32") / 255.0

        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))

        # Convert both to grayscale before subtraction
        original_gray = cv2.cvtColor((original * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        reconstructed_uint8 = (reconstructed * 255).astype("uint8")
        if reconstructed_uint8.ndim == 3 and reconstructed_uint8.shape[2] == 3:
            reconstructed_gray = cv2.cvtColor(reconstructed_uint8, cv2.COLOR_RGB2GRAY)
        else:
            reconstructed_gray = reconstructed_uint8.squeeze()  # remove 3rd dimension if it's (7, 7, 1)
        diff_gray = cv2.absdiff(original_gray, reconstructed_gray)

        norm_diff = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        return overlay

    def take_picture(self):
        if self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                filename = "captured_image.jpg"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Success", f"Picture saved as '{filename}'")

    def toggle_ai(self):
        self.ai_enabled = not self.ai_enabled
        self.ai_toggle.configure(text="Disable AI" if self.ai_enabled else "Enable AI")

    def create_legend(self):
        #"""Creates a vertical heatmap legend with labels."""
        height = 300  # Adjust as needed
        width = 50    # Adjust as needed

        # Create a vertical gradient from blue to red
        gradient = np.linspace(0, 255, height, dtype=np.uint8).reshape((height, 1))
        gradient = np.repeat(gradient, width, axis=1)  # Expand width

        # Apply COLORMAP_JET
        legend_colormap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        legend_colormap = cv2.cvtColor(legend_colormap, cv2.COLOR_BGR2RGB)  # Convert for Tkinter

        # Convert to PIL Image
        legend_image = Image.fromarray(legend_colormap)

        # Add text labels
        draw = ImageDraw.Draw(legend_image)

        try:
            font = ImageFont.truetype("arial.ttf", 14)  # Use "DejaVuSans.ttf" on Linux
        except IOError:
            font = ImageFont.load_default()  # Fallback if font is not found

        draw.text((5, 5), "Low", fill=(255, 255, 255), font=font)
        draw.text((5, height // 2), "Mid", fill=(255, 255, 255), font=font)
        draw.text((5, height - 15), "High", fill=(255, 255, 255), font=font)

        return legend_image

# Run the program
root = ctk.CTk()
app = DetectionProgram(root)
root.mainloop()
