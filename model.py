"""
model.py
--------
Everything related to the face-recognition CNN lives here.

Pipeline overview:
  1. FACE DETECTION   -> OpenCV Haar Cascade locates a face inside a raw photo and
                          crops it out (we don't want to feed the whole background
                          image into the CNN, only the face region).
  2. PREPROCESSING    -> the cropped face is resized to a fixed size (96x96) and its
                          pixel values are normalized. Neural nets train far better on
                          small, consistent, normalized inputs than on raw, variable-size
                          images straight from a camera.
  3. MODEL            -> transfer learning on top of MobileNetV2 (pretrained on
                          ImageNet). We freeze MobileNetV2's convolutional base and
                          train only a small classification "head" on top of it.
  4. TRAINING         -> scans static/dataset/<student_id>/*.jpg, builds arrays, trains
                          the head, and saves the model + a label map to disk.
  5. PREDICTION       -> loads the saved model and predicts which student a new face
                          belongs to, returning a confidence score.

WHY TRANSFER LEARNING INSTEAD OF A CNN FROM SCRATCH?
Face recognition needs rich visual features (edges, textures, facial geometry) to tell
similar-looking faces apart. Training a deep CNN from scratch typically needs thousands
of images per class. A school attendance system realistically has 3-10 photos per
student. MobileNetV2 already learned strong general-purpose visual features from 1.4M
ImageNet images -- we just reuse them and train a small classifier on top. This is the
standard, practical approach for small, custom face datasets.

A plain "few Conv2D + MaxPooling" CNN (trained from scratch) is also included below as
`build_simple_cnn()` for reference / comparison, since it was mentioned as an option.
It will need many more images per student to avoid overfitting.
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import layers, models
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMG_SIZE = 96  # all faces are resized to IMG_SIZE x IMG_SIZE before entering the CNN
DATASET_DIR = os.path.join("static", "dataset")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_cnn.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# OpenCV ships with several pretrained Haar Cascade XML files; this one detects
# frontal (forward-facing) faces, which is what a webcam/attendance photo will show.
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Simple in-memory cache so we don't reload the model from disk on every single
# prediction request (Keras model loading is relatively slow).
_model_cache = {"model": None, "label_map": None}


class TrainingCallback(tf.keras.callbacks.Callback):
    """Custom callback to provide training progress updates"""
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: loss = {logs.get('loss'):.4f}, accuracy = {logs.get('accuracy'):.4f}")


# ---------------------------------------------------------------------------
# Face detection & preprocessing
# ---------------------------------------------------------------------------
def detect_face(image_bgr):
    """
    Detects the largest face in a BGR image (OpenCV's default color order) and
    returns the cropped face region (still BGR). Returns None if no face is found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,      # how much the image size is reduced at each scale step
        minNeighbors=3,       # lower = more detections (including false positives), was 5
        minSize=(50, 50)      # ignore detections smaller than this (was 60, lowered to catch smaller faces)
    )

    if len(faces) == 0:
        # Try with more relaxed parameters if strict detection fails
        faces = _face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # Smaller scale factor for more thorough search
            minNeighbors=2,      # Even lower neighbors requirement
            minSize=(40, 40)     # Even smaller minimum face size
        )

    if len(faces) == 0:
        return None

    # If several faces are detected (e.g. someone walked by in the background),
    # keep only the largest one -- it's almost always the person facing the camera.
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return image_bgr[y:y + h, x:x + w]


def preprocess_face(face_bgr):
    """
    Converts a cropped face (BGR, arbitrary size) into the exact format the CNN
    expects:
      1. Resize      -> IMG_SIZE x IMG_SIZE, so every input has identical dimensions.
      2. Color order  -> BGR (OpenCV) to RGB (TensorFlow/Keras convention).
      3. Normalize    -> MobileNetV2's own `preprocess_input`, which scales pixels
                         from [0, 255] to [-1, 1]. This MUST match how the pretrained
                         weights were originally trained, or the features it extracts
                         will be meaningless.
    Returns a float32 numpy array of shape (IMG_SIZE, IMG_SIZE, 3).
    """
    face_resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_array = face_rgb.astype("float32")
    face_array = preprocess_input(face_array)
    return face_array


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
def build_model(num_classes):
    """
    Transfer-learning model (recommended, used by default):

        MobileNetV2 (frozen, pretrained)  ->  GlobalAveragePooling2D
                                            ->  Dense(128, relu)
                                            ->  Dropout(0.3)
                                            ->  Dense(num_classes, softmax)

    - `include_top=False` strips off MobileNet's original 1000-class ImageNet
      classifier, since we only want its feature extractor.
    - `base_model.trainable = False` freezes those pretrained convolutional
      weights so training doesn't overwrite the useful features already learned,
      and so training is fast even on a CPU with only a handful of images.
    - GlobalAveragePooling2D collapses each feature map down to a single number,
      turning MobileNet's output into a compact feature vector per face.
    - Dropout randomly disables 30% of neurons during training, which helps
      prevent overfitting given how few images per student we likely have.
    - The final Dense(num_classes, softmax) layer outputs one probability per
      registered student, summing to 1.0 (e.g. [0.02, 0.91, 0.07] for 3 students).
    """
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",  # labels are plain integers (0, 1, 2, ...)
        metrics=["accuracy"]
    )
    return model


def build_simple_cnn(num_classes):
    """
    Alternative: a small CNN trained fully from scratch, no pretrained weights.
    Provided for reference / comparison. Only use this if you have a larger
    dataset per student (50+ images), otherwise it will badly overfit.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_dataset():
    """
    Walks static/dataset/<student_id>/*.jpg, detects + preprocesses each face,
    and builds training arrays.

    Returns:
        X          - numpy array of shape (num_images, IMG_SIZE, IMG_SIZE, 3)
        y          - numpy array of integer class labels, one per image in X
        label_map  - dict mapping {class_index: student_id}, e.g. {0: "STU001", 1: "STU002"}
                     This is saved to disk so prediction can translate the CNN's
                     output index back into an actual student_id.
    """
    X, y = [], []
    label_map = {}

    # Only look at actual filesystem folders, not database records
    student_folders = sorted(os.listdir(DATASET_DIR)) if os.path.isdir(DATASET_DIR) else []

    class_index = 0
    for student_id in student_folders:
        folder_path = os.path.join(DATASET_DIR, student_id)
        if not os.path.isdir(folder_path):
            continue

        images_added = 0
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            image_bgr = cv2.imread(filepath)
            if image_bgr is None:
                continue  # skip unreadable / non-image files

            face = detect_face(image_bgr)
            if face is None:
                print(f"Warning: Could not detect face in {filepath}")  # Debug message
                continue  # skip images where no face could be detected

            print(f"Successfully detected face in {filepath}")  # Debug message
            X.append(preprocess_face(face))
            y.append(class_index)
            images_added += 1

        if images_added > 0:
            label_map[class_index] = student_id
            print(f"Student {student_id} has {images_added} usable face images")
            class_index += 1
        else:
            print(f"No usable face images found for student {student_id}")

    if len(X) == 0:
        print("No faces found in any student folders")
        return None, None, {}
    
    print(f"Total: {len(X)} images from {len(label_map)} students")
    return np.array(X, dtype="float32"), np.array(y, dtype="int32"), label_map


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(epochs=10, batch_size=8):
    """
    Trains the CNN on every currently-registered student's images and saves:
      - the trained model   -> models/face_cnn.h5
      - the label map       -> models/labels.json

    Returns a summary dict used by the /train Flask route to report results.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, label_map = load_dataset()
    if X is None:
        return {
            "success": False,
            "message": "No face images found in the dataset directory. Please register students with face images first."
        }
    
    if len(label_map) < 2:
        return {
            "success": False,
            "message": f"Need at least 2 registered students with detectable faces before training. Currently have {len(label_map)}."
        }

    num_classes = len(label_map)
    
    try:
        model = build_model(num_classes)
    except Exception as e:
        return {
            "success": False,
            "message": f"Error building model: {str(e)}"
        }

    # Use a validation split only if we have enough images to make it meaningful.
    val_split = 0.2 if len(X) >= 10 else 0.0

    try:
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            verbose=1,
            callbacks=[TrainingCallback()]  # Add training callback
        )
    except Exception as e:
        return {
            "success": False,
            "message": f"Error during training: {str(e)}"
        }

    try:
        model.save(MODEL_PATH, save_format='h5')  # Explicitly specify format
        with open(LABELS_PATH, "w") as f:
            json.dump(label_map, f)
    except Exception as e:
        return {
            "success": False,
            "message": f"Error saving model: {str(e)}"
        }

    # Clear the cache so the next prediction picks up the freshly trained model.
    _model_cache["model"] = None
    _model_cache["label_map"] = None

    final_acc = history.history.get("accuracy", [0])[-1]
    return {
        "success": True,
        "message": f"Model trained on {len(X)} images across {num_classes} students.",
        "final_accuracy": round(float(final_acc), 4)
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def _get_model_and_labels():
    """Loads the model + label map from disk once, then reuses them from memory."""
    if _model_cache["model"] is None:
        try:
            _model_cache["model"] = tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            _model_cache["model"] = None
            _model_cache["label_map"] = None
            raise e
        
        with open(LABELS_PATH) as f:
            _model_cache["label_map"] = {int(k): v for k, v in json.load(f).items()}
    return _model_cache["model"], _model_cache["label_map"]


def predict_face(image_bgr, confidence_threshold=0.75):
    """
    Given a raw BGR image (e.g. a webcam snapshot or uploaded photo), detects the
    face and predicts which registered student it belongs to.

    Returns a dict:
        {"student_id": "STU001", "confidence": 0.93}                        on a confident match
        {"student_id": None, "confidence": 0.41, "reason": "..."}           otherwise
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return {"student_id": None, "confidence": 0.0, "reason": "Model not trained yet. Visit /train first."}

    face = detect_face(image_bgr)
    if face is None:
        return {"student_id": None, "confidence": 0.0, "reason": "No face detected in image."}

    model, label_map = _get_model_and_labels()

    face_array = preprocess_face(face)
    face_batch = np.expand_dims(face_array, axis=0)  # add the batch dimension Keras expects

    predictions = model.predict(face_batch, verbose=0)[0]  # e.g. [0.02, 0.91, 0.07]
    best_index = int(np.argmax(predictions))
    confidence = float(predictions[best_index])

    if confidence < confidence_threshold:
        return {"student_id": None, "confidence": confidence, "reason": "Confidence below threshold - face not recognized."}

    return {"student_id": label_map.get(best_index), "confidence": confidence}


def diagnose_face_detection():
    """
    Helper function to diagnose face detection issues with current dataset.
    Prints detailed information about which images can/cannot have faces detected.
    """
    print("=== Face Detection Diagnosis ===")
    
    student_folders = sorted(os.listdir(DATASET_DIR)) if os.path.isdir(DATASET_DIR) else []
    
    for student_id in student_folders:
        folder_path = os.path.join(DATASET_DIR, student_id)
        if not os.path.isdir(folder_path):
            continue
            
        print(f"\nStudent: {student_id}")
        
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            image_bgr = cv2.imread(filepath)
            
            if image_bgr is None:
                print(f"  ❌ {filename}: Could not read image")
                continue
                
            face = detect_face(image_bgr)
            if face is None:
                print(f"  ❌ {filename}: No face detected")
            else:
                h, w = face.shape[:2]
                print(f"  ✅ {filename}: Face detected ({w}x{h}px)")
                
    print("\n=== End Diagnosis ===")

