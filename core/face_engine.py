import os
import pickle
import face_recognition
import numpy as np
from PIL import Image

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'encodings.pkl')
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'known_faces')


def load_encodings() -> dict:
    """Load saved encodings from disk. Returns {person_id: {'name': str, 'encoding': ndarray}}"""
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, 'rb') as f:
            return pickle.load(f)
    return {}


def save_encodings(encodings: dict):
    """Persist encodings dictionary to disk."""
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(encodings, f)


def encode_face_from_image(image_path: str):
    """
    Given an image file path, detect and return the first face encoding.
    Returns None if no face found.
    """
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return None
    return encodings[0]


def encode_face_from_array(frame: np.ndarray, is_bgr: bool = False):
    """
    Encode faces from a numpy array (live camera frame).
    Pass is_bgr=True if the array comes from OpenCV (BGR order).
    WebRTC frames should be passed as RGB (is_bgr=False, the default).
    """
    rgb = frame[:, :, ::-1] if is_bgr else frame
    locations = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, locations)
    return locations, encodings


def register_face(person_id: int, name: str, image_path: str) -> bool:
    """
    Encode a face from image and add to the encodings store.
    Returns True on success, False if no face detected.
    """
    encoding = encode_face_from_image(image_path)
    if encoding is None:
        return False

    encodings = load_encodings()
    encodings[person_id] = {'name': name, 'encoding': encoding}
    save_encodings(encodings)
    return True


def remove_face(person_id: int):
    """Remove a person's encoding from the store."""
    encodings = load_encodings()
    if person_id in encodings:
        del encodings[person_id]
        save_encodings(encodings)


def recognize_faces(frame: np.ndarray, tolerance: float = 0.50, is_bgr: bool = False):
    """
    Detect all faces in a frame and match against known encodings.

    frame    : numpy array, RGB by default. Pass is_bgr=True for OpenCV BGR arrays.
    tolerance: match threshold — lower = stricter. 0.50 is a good default.

    Returns a list of dicts:
        [{'location': (top,right,bottom,left), 'person_id': int|None, 'name': str, 'confidence': float}]
    """
    known = load_encodings()
    if not known:
        return []

    known_ids       = list(known.keys())
    known_encodings = [known[pid]['encoding'] for pid in known_ids]
    known_names     = [known[pid]['name']     for pid in known_ids]

    locations, encodings = encode_face_from_array(frame, is_bgr=is_bgr)
    results = []

    for location, encoding in zip(locations, encodings):
        distances   = face_recognition.face_distance(known_encodings, encoding)
        best_idx    = int(np.argmin(distances))
        best_dist   = distances[best_idx]
        confidence  = round((1 - best_dist) * 100, 1)

        if best_dist <= tolerance:
            results.append({
                'location':  location,
                'person_id': known_ids[best_idx],
                'name':      known_names[best_idx],
                'confidence': confidence,
                'matched':   True
            })
        else:
            results.append({
                'location':  location,
                'person_id': None,
                'name':      'Unknown',
                'confidence': confidence,
                'matched':   False
            })

    return results