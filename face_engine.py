"""
face_engine.py
--------------
The face-recognition core, rebuilt around embeddings + similarity matching
instead of a retrained softmax classifier. See the README for the full
rationale; short version:

    OLD: photos -> train a fresh N-class classifier -> predict a class index
         (needs 2+ students, full retrain to add anyone, poor stranger-rejection)

    NEW: photos -> CNN embedding (a 1280-d "fingerprint" vector) -> stored in
         a small gallery file -> new faces are matched by cosine similarity
         against every stored vector
         (works with 1 student, registering someone is instant, strangers are
         rejected by threshold + margin instead of forced into a class)

Pipeline for a single photo:
  1. DETECT   - find the face region. Haar Cascade runs first (fast); if it
                finds nothing, MTCNN (a small CNN-based detector) is tried as
                a fallback since it handles angled/harder faces better.
  2. QUALITY  - reject/warn on faces that are too small, blurry (Laplacian
                variance), or too dark/bright, using OpenCV metrics -- this
                catches bad registration photos before they ever hurt
                recognition accuracy.
  3. PREPROCESS - resize to 96x96, normalize for MobileNetV2.
  4. EMBED    - MobileNetV2 (frozen, ImageNet weights, pooling='avg') maps the
                face to a 1280-d vector. No training involved -- this is pure
                feature extraction, which is why registration is instant.
  5. MATCH    - the new embedding is compared via cosine similarity against
                every embedding in the gallery (static/... no, DATA_DIR/embeddings/
                embeddings.json). The closest student wins IF (a) the
                similarity clears MATCH_THRESHOLD and (b) it beats the
                second-best candidate by at least MATCH_MARGIN -- the margin
                check is what catches "two plausible but wrong" matches that
                a bare threshold would let through.
"""

import os
import json
import numpy as np
import config

try:
    import cv2
    # Check OpenCV version to handle API differences
    opencv_version = cv2.__version__.split('.')[0]
    OPENCV_VERSION_MAJOR = int(opencv_version)
    
    if OPENCV_VERSION_MAJOR >= 5:
        # For OpenCV 5.x, we may need to use alternative approaches
        # Try to access the Haar Cascade data differently
        try:
            # Some installations of OpenCV 5.x may still have the data
            haarcascades_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if hasattr(cv2, 'data') else None
            if haarcascades_path and os.path.exists(haarcascades_path):
                _haar_cascade = cv2.CascadeClassifier(haarcascades_path)
            else:
                # If we can't find the haarcascade file, set to None
                _haar_cascade = None
        except (AttributeError, TypeError):
            # Handle the case where CascadeClassifier is not available or haarcascades data is missing
            _haar_cascade = None
    else:
        # For OpenCV 4.x and earlier
        _haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
except ImportError:
    cv2 = None
    _haar_cascade = None
    print("[face_engine] OpenCV not available - install with 'pip install opencv-python==4.8.1.78'")

_mtcnn_detector = None


def _get_mtcnn():
    global _mtcnn_detector
    if _mtcnn_detector is None:
        try:
            from mtcnn import MTCNN
            # Initialize MTCNN with default parameters (MTCNN doesn't support min_face_size parameter)
            _mtcnn_detector = MTCNN()
        except ImportError:
            print("[face_engine] MTCNN not available - install with 'pip install mtcnn'")
            return None
    return _mtcnn_detector


def detect_face(image_bgr):
    """
    Finds the largest/most confident face in a BGR image and returns the
    cropped face region (BGR). Returns None if no face could be found by
    either detector.
    """
    if cv2 is None:
        print("[face_engine] OpenCV not available, cannot detect faces")
        return None

    # If Haar Cascade is not available (OpenCV 5.x issue), go straight to MTCNN
    if _haar_cascade is None:
        print("[face_engine] Haar Cascade not available, using MTCNN exclusively")
        # Use MTCNN as primary detector if OpenCV Haar is unavailable
        mtcnn = _get_mtcnn()
        if mtcnn is not None:
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                results = mtcnn.detect_faces(image_rgb)
                if results:
                    # Find the largest face instead of just the most confident
                    best = max(results, key=lambda r: r["box"][2] * r["box"][3])  # width * height
                    x, y, w, h = best["box"]
                    x, y = max(0, x), max(0, y)
                    face = image_bgr[y:y + h, x:x + w]
                    if face.size > 0:
                        return face
            except Exception as e:
                print(f"[face_engine] MTCNN detection failed: {e}")
        return None
        
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Use more relaxed parameters for face detection to catch more faces in uploaded images
        faces = _haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

        if len(faces) > 0:
            # Select the largest face (most likely to be the main subject)
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            return image_bgr[y:y + h, x:x + w]

        # Haar found nothing -- try MTCNN, which copes much better with angled
        # faces, partial occlusion, and tricky lighting (at the cost of being
        # slower, which is fine for a one-shot attendance check).
        mtcnn = _get_mtcnn()
        if mtcnn is not None:
            try:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                results = mtcnn.detect_faces(image_rgb)
                if results:
                    # Find the largest face instead of just the most confident
                    best = max(results, key=lambda r: r["box"][2] * r["box"][3])  # width * height
                    x, y, w, h = best["box"]
                    x, y = max(0, x), max(0, y)
                    face = image_bgr[y:y + h, x:x + w]
                    if face.size > 0:
                        return face
            except Exception as e:
                print(f"[face_engine] MTCNN fallback failed: {e}")
        else:
            print("[face_engine] MTCNN not available, skipping fallback detection")

    except Exception as e:
        print(f"[face_engine] Error in detect_face: {e}")
        return None

    return None


def assess_quality(face_bgr):
    """
    Runs cheap, fast heuristics on a cropped face and returns a list of
    human-readable warning strings (empty list = looks good). These are
    used to give the person live feedback while registering/checking in,
    rather than silently accepting a bad photo that hurts accuracy later.
    """
    warnings = []
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    if min(h, w) < config.MIN_FACE_SIZE:
        warnings.append(f"Face looks small in frame ({w}x{h}px) — try moving closer.")

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < config.BLUR_THRESHOLD:
        warnings.append("Image looks blurry — hold still and make sure the camera is focused.")

    brightness = float(np.mean(gray))
    if brightness < config.MIN_BRIGHTNESS:
        warnings.append("Image is quite dark — try better lighting.")
    elif brightness > config.MAX_BRIGHTNESS:
        warnings.append("Image is overexposed — reduce direct light or glare.")

    return warnings


# ---------------------------------------------------------------------------
# Preprocessing + embedding extraction
# ---------------------------------------------------------------------------
def preprocess_face(face_bgr):
    """Resize to IMG_SIZE and normalize the way MobileNetV2 expects (pixels -> [-1, 1])."""
    from keras.applications.mobilenet_v2 import preprocess_input

    if face_bgr is None:
        raise ValueError("preprocess_face(): received None")

    if not isinstance(face_bgr, np.ndarray):
        raise TypeError(f"preprocess_face(): expected numpy array, got {type(face_bgr)}")

    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError(f"preprocess_face(): expected HxWx3 BGR image, got shape {face_bgr.shape}")

    h, w = face_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"preprocess_face(): empty crop with shape {face_bgr.shape}")

    # OpenCV can throw on extremely small/invalid inputs; guard with sane bounds.
    if h < 2 or w < 2:
        raise ValueError(f"preprocess_face(): crop too small with shape {face_bgr.shape}")

    face_resized = cv2.resize(face_bgr, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    if face_resized.shape[0] != config.IMG_SIZE or face_resized.shape[1] != config.IMG_SIZE:
        raise ValueError(f"preprocess_face(): resize produced unexpected shape {face_resized.shape}")

    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Add explicit batch dimension before preprocessing
    face_rgb = np.expand_dims(face_rgb, axis=0)
    
    out = preprocess_input(face_rgb)

    if not (isinstance(out, np.ndarray) and out.shape == (1, config.IMG_SIZE, config.IMG_SIZE, 3)):
        raise ValueError(f"preprocess_face(): unexpected output shape {getattr(out, 'shape', None)}")

    return out


_embedder = None


def _get_embedder():
    """
    Lazy-loads MobileNetV2 as a pure feature extractor. `pooling='avg'` adds
    a GlobalAveragePooling2D on top of the conv base, so calling .predict()
    directly returns a (batch, 1280) embedding -- no extra layers or training
    required. Weights are frozen ImageNet weights; we never fine-tune them,
    which is exactly why there's no training step in this architecture.
    """
    global _embedder
    if _embedder is None:
        from keras.applications.mobilenet_v2 import MobileNetV2
        # Disable TensorFlow optimizations during model creation to speed things up
        import tensorflow as tf
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        
        _embedder = MobileNetV2(
            input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        # Warm up the model by running a dummy prediction
        dummy_input = np.random.rand(1, config.IMG_SIZE, config.IMG_SIZE, 3).astype('float32')
        _embedder.predict(dummy_input, verbose=0)
    return _embedder


def warm_up_models():
    """
    Warm up all models at startup to reduce latency during first use.
    """
    print("[face_engine] Warming up models...")
    
    # Initialize the embedder model
    embedder = _get_embedder()
    
    # Initialize MTCNN detector
    mtcnn = _get_mtcnn()
    
    print("[face_engine] Models warmed up successfully!")


# Don't call warm_up_models at module level to avoid import issues
# warm_up_models()


def compute_embedding(face_bgr):
    """Returns an L2-normalized 1280-d embedding vector for a cropped face."""
    embedder = _get_embedder()
    face_array = preprocess_face(face_bgr)

    # The face_array already has the batch dimension from the preprocessing fix
    batch = face_array
    
    # Defensive: ensure batch rank/shape is what the model expects.
    # Model input is (None, IMG_SIZE, IMG_SIZE, 3)
    if batch.ndim != 4 or batch.shape[1:] != (config.IMG_SIZE, config.IMG_SIZE, 3):
        raise ValueError(f"compute_embedding(): unexpected batch shape {batch.shape}")

    raw = embedder.predict(batch, verbose=0)[0]
    norm = np.linalg.norm(raw)
    return (raw / norm) if norm > 0 else raw



def cosine_similarity(a, b):
    """Dot product of two already-L2-normalized vectors == cosine similarity, in [-1, 1]."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Gallery (student_id -> list of embedding vectors) persistence
# ---------------------------------------------------------------------------
def load_gallery():
    """Returns {student_id: [embedding, embedding, ...]} as numpy arrays."""
    if not os.path.exists(config.EMBEDDINGS_PATH):
        return {}
    with open(config.EMBEDDINGS_PATH) as f:
        raw = json.load(f)
    return {sid: [np.array(e, dtype="float32") for e in embeddings] for sid, embeddings in raw.items()}


def save_gallery(gallery):
    config.ensure_directories()
    serializable = {sid: [e.tolist() for e in embeddings] for sid, embeddings in gallery.items()}
    with open(config.EMBEDDINGS_PATH, "w") as f:
        json.dump(serializable, f)


def add_photo_to_gallery(student_id, face_bgr):
    """Computes and stores one more reference embedding for a student."""
    embedding = compute_embedding(face_bgr)
    gallery = load_gallery()
    gallery.setdefault(student_id, []).append(embedding)
    save_gallery(gallery)
    return embedding


def remove_student_from_gallery(student_id):
    gallery = load_gallery()
    if student_id in gallery:
        del gallery[student_id]
        save_gallery(gallery)


def reindex_gallery():
    """
    Rebuilds embeddings.json from scratch by re-reading every photo under
    DATASET_DIR/<student_id>/*. Useful after bulk-importing photos directly
    onto disk, or if the embedding model ever changes and old vectors need
    recomputing.
    """
    gallery = {}
    processed_images = 0
    skipped_images = 0

    if os.path.isdir(config.DATASET_DIR):
        for student_id in sorted(os.listdir(config.DATASET_DIR)):
            folder = os.path.join(config.DATASET_DIR, student_id)
            if not os.path.isdir(folder):
                continue

            embeddings = []
            try:
                for filename in sorted(os.listdir(folder)):
                    filepath = os.path.join(folder, filename)
                    try:
                        image_bgr = cv2.imread(filepath)
                        if image_bgr is None:
                            skipped_images += 1
                            continue
                        face = detect_face(image_bgr)
                        if face is None:
                            skipped_images += 1
                            continue
                        embeddings.append(compute_embedding(face))
                        processed_images += 1
                    except Exception as e:
                        print(f"[face_engine] Error processing image {filepath}: {e}")
                        skipped_images += 1
            except PermissionError:
                print(f"[face_engine] Permission denied accessing folder {folder}")
                continue

            if embeddings:
                gallery[student_id] = embeddings

    save_gallery(gallery)
    return {
        "students_indexed": len(gallery),
        "images_processed": processed_images,
        "images_skipped": skipped_images
    }


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
def match_face(image_bgr):
    """
    Full pipeline for an attendance check: detect -> embed -> compare against
    the gallery.

    Returns a dict:
        {"student_id": "STU001", "confidence": 0.81}                    on a confident, unambiguous match
        {"student_id": None, "confidence": 0.0, "reason": "..."}        otherwise
    """
    face = detect_face(image_bgr)
    if face is None:
        print(f"[face_engine] No face detected in image of size {image_bgr.shape if image_bgr is not None else 'None'}")
        return {"student_id": None, "confidence": 0.0, "reason": "No face detected in the image."}

    gallery = load_gallery()
    if not gallery:
        return {"student_id": None, "confidence": 0.0, "reason": "No students registered yet."}

    try:
        query_embedding = compute_embedding(face)
    except Exception as e:
        # If MTCNN produced a degenerate crop, preprocessing/embedding may fail.
        # Treat as "no usable face" rather than crashing/throwing invalid
        # tensors through Keras.
        print(f"[face_engine] Failed to compute embedding: {e}")
        return {"student_id": None, "confidence": 0.0, "reason": f"Could not compute face embedding: {e}"}


    # Use numpy for faster computation of similarities
    student_ids = list(gallery.keys())
    all_embeddings = []
    student_indices = []
    
    for idx, student_id in enumerate(student_ids):
        embeddings = gallery[student_id]
        for embedding in embeddings:
            all_embeddings.append(embedding)
            student_indices.append(idx)
    
    if not all_embeddings:
        return {"student_id": None, "confidence": 0.0, "reason": "No valid face embeddings to compare against."}
    
    all_embeddings = np.array(all_embeddings)
    query_embedding = np.array(query_embedding)
    
    # Compute similarities in batch
    similarities = np.dot(all_embeddings, query_embedding)
    
    # Group by student and find best score per student
    best_per_student = {}
    for i, student_idx in enumerate(student_indices):
        student_id = student_ids[student_idx]
        similarity = similarities[i]
        if student_id not in best_per_student or best_per_student[student_id] < similarity:
            best_per_student[student_id] = similarity

    ranked = sorted(best_per_student.items(), key=lambda item: item[1], reverse=True)
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0

    if best_score < config.MATCH_THRESHOLD:
        return {"student_id": None, "confidence": best_score, "reason": f"Face not recognized — confidence {best_score:.2f} is below threshold {config.MATCH_THRESHOLD:.2f}."}

    if (best_score - second_score) < config.MATCH_MARGIN and len(ranked) > 1:
        return {"student_id": None, "confidence": best_score,
                "reason": "Match too close between two students — please retake the photo."}

    return {"student_id": best_id, "confidence": best_score}