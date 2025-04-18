import tkinter as tk
from tkinter import filedialog, ttk, Text # Import Text widget
from PIL import Image, ImageTk
import torch
import numpy as np
import cv2 # Use opencv for image handling, colormapping, and privacy filter
from transformers import DPTImageProcessor, DPTForDepthEstimation
import threading # Import threading for non-blocking model load
import os # Import os to locate cascade files

# --- Global Variables ---
model = None
processor = None
original_image_tk = None
depth_map_tk = None
privacy_filtered_tk = None # New global for privacy image
original_img_label = None
depth_map_label = None
privacy_filter_label = None # New global for privacy label
status_label = None
root = None
browse_button = None # Add browse_button to globals to disable/enable it
poses_text = None # Global for poses text box
intrinsics_text = None # Global for intrinsics text box
MAX_DISPLAY_WIDTH = 350 # Adjusted slightly for 3 images
MAX_DISPLAY_HEIGHT = 350 # Adjusted slightly for 3 images
MODEL_NAME = "Intel/dpt-hybrid-midas" # Define model name constant

# --- Haar Cascade Classifiers (Load once if possible, but safer to load in function for now) ---
# Locate Haar cascade files provided by OpenCV
haar_base_path = cv2.data.haarcascades
face_cascade_path = os.path.join(haar_base_path, 'haarcascade_frontalface_default.xml')
# Note: OpenCV's built-in license plate detectors are less reliable than face detectors.
# You might need a more specialized model for robust plate detection.
# Example using one of the available (often Russian focused) plate cascades:
# plate_cascade_path = os.path.join(haar_base_path, 'haarcascade_russian_plate_number.xml')
plate_cascade_path = os.path.join(haar_base_path, 'haarcascade_license_plate_rus_16stages.xml') # Another option

face_cascade = None
plate_cascade = None

# --- Load Cascades ---
# It's generally better to load these once, but for simplicity in this script,
# we'll load them within the filter function. If performance is critical,
# load them globally after checking existence.
def load_cascades():
    global face_cascade, plate_cascade
    if os.path.exists(face_cascade_path):
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        print("Face cascade loaded.")
    else:
        print(f"Error: Face cascade file not found at {face_cascade_path}")
        face_cascade = None

    if os.path.exists(plate_cascade_path):
        plate_cascade = cv2.CascadeClassifier(plate_cascade_path)
        print("License plate cascade loaded.")
    else:
        print(f"Warning: License plate cascade file not found at {plate_cascade_path}")
        plate_cascade = None # Continue without plate detection if not found

# --- Depth Estimation Logic ---
# (Keep existing functions: load_model_worker, start_model_load, estimate_depth, apply_colormap)
def load_model_worker():
    """Worker function to load the model in a separate thread."""
    global model, processor, status_label, browse_button, root
    try:
        print(f"Loading model: {MODEL_NAME}...")
        # Use root.after to safely update GUI from the worker thread
        root.after(0, lambda: status_label.config(text=f"Loading model ({MODEL_NAME.split('/')[-1]})..."))
        root.after(0, lambda: browse_button.config(state=tk.DISABLED)) # Disable button during load

        processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
        model = DPTForDepthEstimation.from_pretrained(MODEL_NAME)

        print("Model loaded successfully.")
        # Use root.after to safely update GUI from the worker thread
        root.after(0, lambda: status_label.config(text=f"Model loaded ({MODEL_NAME.split('/')[-1]}). Ready."))
        root.after(0, lambda: browse_button.config(state=tk.NORMAL)) # Re-enable button

        # Load cascades after model is loaded (or could be done earlier)
        load_cascades() # Load face/plate detectors

    except Exception as e:
        error_msg = f"Error loading model: {e}"
        print(error_msg)
        if status_label and root:
             # Use root.after to safely update GUI from the worker thread
            root.after(0, lambda: status_label.config(text=error_msg))
            # Keep button disabled if model load failed
            root.after(0, lambda: browse_button.config(state=tk.DISABLED))

def start_model_load():
    """Starts loading the model in a background thread."""
    # Create and start a new thread for loading the model
    thread = threading.Thread(target=load_model_worker, daemon=True) # daemon=True allows closing app even if thread runs
    thread.start()


def estimate_depth(image_path):
    """Estimates depth for a given image path."""
    global model, processor
    if model is None or processor is None:
        print("Model not loaded.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")

        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1], # PIL size is (width, height), tensor size is (height, width)
            mode="bicubic",
            align_corners=False,
        )

        # Normalize and format output
        output = prediction.squeeze().cpu().numpy()
        # Ensure non-zero max value to avoid division by zero
        max_val = np.max(output)
        if max_val == 0:
            # Handle case of all-black depth map (though unlikely)
             formatted = np.zeros_like(output, dtype=np.uint8)
        else:
            formatted = (output * 255 / max_val).astype("uint8")

        depth_map_image = Image.fromarray(formatted)

        return image, depth_map_image

    except Exception as e:
        print(f"Error during depth estimation: {e}")
        if status_label:
            status_label.config(text=f"Error processing image: {e}")
        return None, None

def apply_colormap(depth_image_pil):
    """Applies a colormap to the grayscale depth image."""
    if depth_image_pil is None:
        return None
    depth_cv = np.array(depth_image_pil)
    # Invert the depth map before colormapping if desired (closer = brighter/hotter color)
    # depth_cv = 255 - depth_cv # Uncomment this line to invert depth perception
    colored_depth = cv2.applyColorMap(depth_cv, cv2.COLORMAP_MAGMA) # Changed colormap
    # Convert BGR (OpenCV default) to RGB (PIL/Tkinter default)
    colored_depth_rgb = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_depth_rgb)

# --- Privacy Filter Logic ---
def apply_privacy_filter(image_pil):
    """Applies blurring to detected faces and license plates."""
    global face_cascade, plate_cascade

    if image_pil is None:
        return None

    if face_cascade is None and plate_cascade is None:
        print("No cascades loaded for privacy filter. Returning original image.")
        return image_pil # Return original if no detectors are available

    try:
        # Convert PIL Image to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        # Create a copy to draw on, preserving the original cv image if needed elsewhere
        output_image_cv = image_cv.copy()
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # --- Detect and Blur Faces ---
        if face_cascade:
            # Adjust parameters as needed: scaleFactor, minNeighbors, minSize
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            print(f"Detected {len(faces)} faces.")
            for (x, y, w, h) in faces:
                # Extract the region of interest (ROI) - the face
                face_roi = output_image_cv[y:y+h, x:x+w]
                # Apply Gaussian blur - kernel size must be odd
                # Increase kernel size or sigma for more blur
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                # Replace the original face ROI with the blurred version
                output_image_cv[y:y+h, x:x+w] = blurred_face

        # --- Detect and Blur License Plates ---
        if plate_cascade:
            # Plate detection is often less reliable, parameters might need tuning
            plates = plate_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4, # May need lower neighbors for plates
                minSize=(40, 15) # Adjust size based on expected plate dimensions
            )
            print(f"Detected {len(plates)} potential license plates.")
            for (x, y, w, h) in plates:
                # Extract ROI
                plate_roi = output_image_cv[y:y+h, x:x+w]
                # Apply blur (can use different kernel/sigma if desired)
                blurred_plate = cv2.GaussianBlur(plate_roi, (33, 33), 15)
                # Replace ROI
                output_image_cv[y:y+h, x:x+w] = blurred_plate

        # Convert the processed OpenCV image (BGR) back to PIL format (RGB)
        privacy_filtered_pil = Image.fromarray(cv2.cvtColor(output_image_cv, cv2.COLOR_BGR2RGB))
        return privacy_filtered_pil

    except Exception as e:
        print(f"Error during privacy filtering: {e}")
        # Return the original image in case of error during processing
        return image_pil


# --- GUI Logic ---
def resize_image_for_display(img):
    """Resizes a PIL image to fit the display area while maintaining aspect ratio."""
    if img is None:
        return None
    w, h = img.size
    # Use local max dimensions if possible, otherwise fallback to global
    max_w = MAX_DISPLAY_WIDTH
    max_h = MAX_DISPLAY_HEIGHT

    if w > max_w or h > max_h:
        ratio = min(max_w / w, max_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        # Use ANTIALIAS for older PIL, LANCZOS for newer Pillow
        resample_method = Image.Resampling.LANCZOS if hasattr(Image.Resampling, 'LANCZOS') else Image.ANTIALIAS
        try:
            img = img.resize((new_w, new_h), resample_method)
        except Exception as e:
             print(f"Error resizing image: {e}")
             return None # Return None if resize fails
    return img


def open_image():
    """Opens a file dialog and processes the selected image."""
    global original_image_tk, depth_map_tk, privacy_filtered_tk # Add privacy tk
    global original_img_label, depth_map_label, privacy_filter_label # Add privacy label
    global status_label, browse_button
    # Also access text boxes if you want to update them, e.g., with file info (optional)
    # global poses_text, intrinsics_text

    if model is None or processor is None:
        status_label.config(text="Model not loaded yet. Please wait or check for errors.")
        return
    if face_cascade is None and plate_cascade is None:
         status_label.config(text="Warning: Privacy cascades not loaded. Filtering may not occur.")
         # Allow proceeding, but filtering won't happen

    # Temporarily disable button during processing
    browse_button.config(state=tk.DISABLED)
    root.update_idletasks()

    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"), ("All files", "*.*")]
    )

    if not file_path:
        browse_button.config(state=tk.NORMAL) # Re-enable if cancelled
        return # User cancelled

    status_label.config(text=f"Processing {file_path.split('/')[-1]}...")
    root.update_idletasks()

    # --- Run estimation and filtering in a separate thread ---
    def process_worker():
        # 1. Estimate Depth
        original_img_pil, depth_map_pil = estimate_depth(file_path)

        # 2. Apply Privacy Filter (if original image loaded successfully)
        privacy_filtered_pil = None
        if original_img_pil:
            privacy_filtered_pil = apply_privacy_filter(original_img_pil)

        # Update GUI from the main thread using root.after
        def update_gui_callback():
            global original_image_tk, depth_map_tk, privacy_filtered_tk # Need to declare global again

            # Clear previous images first
            original_img_label.config(image='', text="Processing...")
            depth_map_label.config(image='', text="Processing...")
            privacy_filter_label.config(image='', text="Processing...")
            original_img_label.image = None
            depth_map_label.image = None
            privacy_filter_label.image = None
            root.update_idletasks() # Refresh display

            processing_successful = False
            if original_img_pil:
                # Display Original Image
                original_img_pil_resized = resize_image_for_display(original_img_pil)
                if original_img_pil_resized:
                    original_image_tk = ImageTk.PhotoImage(original_img_pil_resized)
                    original_img_label.config(image=original_image_tk, text="") # Clear placeholder text
                    original_img_label.image = original_image_tk # Keep ref
                    processing_successful = True # At least original worked

                # Display Depth Map (if created)
                if depth_map_pil:
                    colored_depth_map_pil = apply_colormap(depth_map_pil)
                    colored_depth_map_pil_resized = resize_image_for_display(colored_depth_map_pil)
                    if colored_depth_map_pil_resized:
                        depth_map_tk = ImageTk.PhotoImage(colored_depth_map_pil_resized)
                        depth_map_label.config(image=depth_map_tk, text="")
                        depth_map_label.image = depth_map_tk # Keep ref

                # Display Privacy Filtered Image (if created)
                if privacy_filtered_pil:
                    privacy_filtered_pil_resized = resize_image_for_display(privacy_filtered_pil)
                    if privacy_filtered_pil_resized:
                        privacy_filtered_tk = ImageTk.PhotoImage(privacy_filtered_pil_resized)
                        privacy_filter_label.config(image=privacy_filtered_tk, text="")
                        privacy_filter_label.image = privacy_filtered_tk # Keep ref
                    else:
                        privacy_filter_label.config(text="Resize failed", image='')
                else:
                     privacy_filter_label.config(text="Filter failed", image='')


            # Update status and handle failures
            if processing_successful:
                 status_label.config(text="Processing complete.")
                 # --- Example: Update text boxes --- (Optional)
                 # poses_text.config(state=tk.NORMAL) ... etc
            else:
                 # Clear all images if primary processing failed
                 original_img_label.config(image='', text="Load an image")
                 depth_map_label.config(image='', text="Depth map will appear here")
                 privacy_filter_label.config(image='', text="Privacy filter will appear here")
                 original_img_label.image = None
                 depth_map_label.image = None
                 privacy_filter_label.image = None
                 if not status_label.cget("text").startswith("Error"): # Don't overwrite specific errors
                     status_label.config(text="Failed to process image.")

            browse_button.config(state=tk.NORMAL) # Re-enable button after processing

        root.after(0, update_gui_callback) # Schedule GUI update

    # Start the processing thread
    process_thread = threading.Thread(target=process_worker, daemon=True)
    process_thread.start()


def create_gui():
    """Creates the main Tkinter window and widgets."""
    global original_img_label, depth_map_label, privacy_filter_label # Add privacy label
    global status_label, root, browse_button
    global poses_text, intrinsics_text # Make text boxes global

    root = tk.Tk()
    # Use model name in title
    root.title(f"Geospatial Imaging Demo ({MODEL_NAME.split('/')[-1]}) + Privacy Filter")
    root.geometry("1200x700") # Increased width for the third image

    # Configure grid weights for resizing
    root.columnconfigure(0, weight=1) # Let the single main column expand
    root.rowconfigure(0, weight=0) # Row 0: Controls frame (no expansion)
    root.rowconfigure(1, weight=1) # Row 1: Image frame (primary vertical expansion)
    root.rowconfigure(2, weight=0) # Row 2: Text frame (fixed height initially)

    # --- Top Controls ---
    control_frame = ttk.Frame(root, padding="10")
    control_frame.grid(row=0, column=0, sticky="ew")

    browse_button = ttk.Button(control_frame, text="Open Image", command=open_image, state=tk.DISABLED) # Start disabled
    browse_button.pack(side=tk.LEFT, padx=5)

    status_label = ttk.Label(control_frame, text="Initializing...")
    status_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    # --- Image Display Area ---
    image_frame = ttk.Frame(root, padding="10")
    image_frame.grid(row=1, column=0, sticky="nsew")
    # Configure columns *within* the image frame for THREE images
    image_frame.columnconfigure(0, weight=1)
    image_frame.columnconfigure(1, weight=1)
    image_frame.columnconfigure(2, weight=1) # Add third column weight
    image_frame.rowconfigure(1, weight=1) # Let image labels expand vertically within frame

    # Labels for image titles
    ttk.Label(image_frame, text="Original Image", font=("Arial", 12)).grid(row=0, column=0, pady=(0, 5))
    ttk.Label(image_frame, text="Depth Map (Colored)", font=("Arial", 12)).grid(row=0, column=1, pady=(0, 5))
    ttk.Label(image_frame, text="Privacy Filter", font=("Arial", 12)).grid(row=0, column=2, pady=(0, 5)) # New title label

    # Labels to display images (using Label with background for placeholder)
    original_img_label = ttk.Label(image_frame, borderwidth=1, relief="solid", background="#dddddd", anchor=tk.CENTER)
    original_img_label.grid(row=1, column=0, padx=5, pady=5, sticky="nsew") # Reduced padx slightly

    depth_map_label = ttk.Label(image_frame, borderwidth=1, relief="solid", background="#dddddd", anchor=tk.CENTER)
    depth_map_label.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    privacy_filter_label = ttk.Label(image_frame, borderwidth=1, relief="solid", background="#dddddd", anchor=tk.CENTER) # New image label
    privacy_filter_label.grid(row=1, column=2, padx=5, pady=5, sticky="nsew") # Grid new label

    # Add placeholder text (optional)
    original_img_label.config(text="Load an image")
    depth_map_label.config(text="Depth map will appear here")
    privacy_filter_label.config(text="Privacy filter will appear here") # Placeholder

    # --- Text Display Area ---
    # (Keep existing text area setup)
    text_frame = ttk.Frame(root, padding=(10, 0, 10, 10)) # Padding: left, top, right, bottom
    text_frame.grid(row=2, column=0, sticky="ew") # Span the single main column, stick horizontally
    text_frame.columnconfigure(0, weight=1)
    text_frame.columnconfigure(1, weight=1)
    text_frame.rowconfigure(1, weight=1)

    ttk.Label(text_frame, text="Poses(IMU Data)", font=("Arial", 12)).grid(row=0, column=0, pady=(5, 2), sticky='w', padx=5)
    ttk.Label(text_frame, text="Camera Intrinsics", font=("Arial", 12)).grid(row=0, column=1, pady=(5, 2), sticky='w', padx=5)

    poses_text_frame = ttk.Frame(text_frame)
    poses_text_frame.grid(row=1, column=0, padx=(5, 2), pady=2, sticky="nsew")
    poses_text_frame.rowconfigure(0, weight=1)
    poses_text_frame.columnconfigure(0, weight=1)

    poses_text = Text(poses_text_frame, height=8, width=40, wrap=tk.WORD, borderwidth=1, relief="solid", font=("Courier New", 9))
    poses_scrollbar = ttk.Scrollbar(poses_text_frame, orient=tk.VERTICAL, command=poses_text.yview)
    poses_text['yscrollcommand'] = poses_scrollbar.set
    poses_text.grid(row=0, column=0, sticky="nsew")
    poses_scrollbar.grid(row=0, column=1, sticky="ns")

    intrinsics_text_frame = ttk.Frame(text_frame)
    intrinsics_text_frame.grid(row=1, column=1, padx=(2, 5), pady=2, sticky="nsew")
    intrinsics_text_frame.rowconfigure(0, weight=1)
    intrinsics_text_frame.columnconfigure(0, weight=1)

    intrinsics_text = Text(intrinsics_text_frame, height=8, width=40, wrap=tk.WORD, borderwidth=1, relief="solid", font=("Courier New", 9))
    intrinsics_scrollbar = ttk.Scrollbar(intrinsics_text_frame, orient=tk.VERTICAL, command=intrinsics_text.yview)
    intrinsics_text['yscrollcommand'] = intrinsics_scrollbar.set
    intrinsics_text.grid(row=0, column=0, sticky="nsew")
    intrinsics_scrollbar.grid(row=0, column=1, sticky="ns")

    # --- Start Model Load ---
    root.after(100, start_model_load) # Start loading model + cascades

    root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure OpenCV is installed: pip install opencv-python
    # Haar cascades should be included with the package.
    create_gui()