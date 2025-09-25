import cv2
import numpy as np
import tensorflow as tf
import time
import os

# ------------------------------
# Config
# ------------------------------
IMG_SIZE = (224, 224)
TRAIN_DIR = "D:\Engineering\SEM07\DIGITAL IMAGE PROCESSING\Project\models\garbage_classification"  # path to your training folder with class subfolders

# ------------------------------
# Load class names automatically
# ------------------------------
classes = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
class_to_idx = {cls: i for i, cls in enumerate(classes)}
print("✅ Classes found:", classes)

# ------------------------------
# Preprocessing (same as training)
# ------------------------------
def largest_object_crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 5)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    crop = bgr[y:y+h, x:x+w]
    return crop if crop.size else bgr

def preprocess_for_model(bgr):
    crop = largest_object_crop(bgr)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, IMG_SIZE)
    x = rgb.astype(np.float32) / 255.0
    return x

# ------------------------------
# TFLite setup
# ------------------------------
tflite_model_path = "model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("✅ TFLite model loaded")
print("Input details:", input_details)
print("Output details:", output_details)

def preprocess_for_tflite(frame_bgr):
    x = preprocess_for_model(frame_bgr)  # float32 [0,1]
    dtype = np.dtype(input_details["dtype"])
    q_scale, q_zero_point = input_details.get("quantization", (0.0, 0))
    if np.issubdtype(dtype, np.floating):
        return np.expand_dims(x.astype(np.float32), axis=0)
    else:
        # int8/uint8 quantization
        if q_scale == 0:
            x_q = np.clip(np.round(x * 255.0), 0, 255).astype(dtype)
        else:
            x_q = np.round(x / q_scale + q_zero_point).astype(dtype)
        return np.expand_dims(x_q, axis=0)

def tflite_output_to_probs(raw_out):
    o_scale, o_zp = output_details.get("quantization", (0.0, 0))
    out = (raw_out.astype(np.float32) - o_zp) * o_scale if o_scale else raw_out.astype(np.float32)
    if np.any(out < 0) or not (0.9 <= np.sum(out) <= 1.1):
        out = tf.nn.softmax(out).numpy()
    return out

# ------------------------------
# Webcam loop
# ------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

prev_time = time.time()
fps_smooth = None

print("Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess and predict
    x = preprocess_for_tflite(frame)
    interpreter.set_tensor(input_details["index"], x)
    interpreter.invoke()
    raw_out = interpreter.get_tensor(output_details["index"])[0]
    probs = tflite_output_to_probs(raw_out)
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    class_name = classes[class_id]

    # FPS smoothing
    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = 1.0 / dt if dt>0 else 0
    fps_smooth = fps if fps_smooth is None else fps_smooth*0.85 + fps*0.15

    # Display
    text = f"{class_name} ({confidence:.2f})  FPS:{fps_smooth:.1f}"
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0), 2)
    cv2.imshow("Webcam - TFLite Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
