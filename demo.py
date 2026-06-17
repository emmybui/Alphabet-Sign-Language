import argparse
import json
import time
import threading
import queue
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


EPS = 1e-6
MIDDLE_FINGER_MCP_INDEX = 9
NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS

# Model hand landmarker chính thức của MediaPipe Tasks.
DEFAULT_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_HAND_LANDMARKER_TASK = "hand_landmarker.task"

# Kết nối giữa các landmarks để vẽ skeleton tay mà không cần mp.solutions.drawing_utils.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def ensure_hand_landmarker_model(task_path: Path, url: str = DEFAULT_HAND_LANDMARKER_URL) -> Path:
    """Tải hand_landmarker.task nếu local chưa có file này."""
    if task_path.exists() and task_path.stat().st_size > 0:
        return task_path

    print(f"[INFO] Không tìm thấy {task_path}. Đang tải MediaPipe hand_landmarker.task...")
    try:
        urllib.request.urlretrieve(url, task_path)
    except Exception as exc:
        raise RuntimeError(
            "Không tải được hand_landmarker.task. Hãy tải thủ công từ URL trong code "
            "hoặc truyền --hand_landmarker_task tới file .task đã có."
        ) from exc

    if not task_path.exists() or task_path.stat().st_size == 0:
        raise RuntimeError("File hand_landmarker.task tải về không hợp lệ.")

    return task_path


def load_label_map(label_map_path: str):
    """Đọc label_map.json. Hỗ trợ cả format mới và format đơn giản {"0": "A"}."""
    with open(label_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "id_to_label" in data:
        id_to_label = {int(k): v for k, v in data["id_to_label"].items()}
    else:
        id_to_label = {int(k): v for k, v in data.items()}

    return id_to_label


def landmarks_to_array(hand_landmarks):
    """
    Chuyển output landmarks về np.ndarray shape (21, 3).
    Hỗ trợ MediaPipe Tasks: result.hand_landmarks[0] là list 21 landmarks.
    """
    if hand_landmarks is None:
        return None

    # MediaPipe legacy mp.solutions có .landmark; MediaPipe Tasks trả list trực tiếp.
    if hasattr(hand_landmarks, "landmark"):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks

    if len(landmarks) != NUM_LANDMARKS:
        return None

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    return coords


def normalize_landmarks(hand_landmarks):
    """
    Normalize landmarks giống hệt lúc training:
    1. Lấy 21 điểm x, y, z.
    2. Trừ tọa độ wrist landmark 0 để đưa wrist về gốc.
    3. Scale theo khoảng cách wrist -> middle finger MCP landmark 9.
    4. Flatten thành vector 63 features.
    """
    coords = landmarks_to_array(hand_landmarks)
    if coords is None or coords.shape != (NUM_LANDMARKS, NUM_COORDS):
        return None

    wrist = coords[0].copy()
    coords = coords - wrist

    scale = np.linalg.norm(coords[MIDDLE_FINGER_MCP_INDEX])
    if scale < EPS:
        return None

    coords = coords / scale
    features = coords.flatten().astype(np.float32)

    if features.shape[0] != FEATURE_DIM:
        return None

    return features


def draw_hand_landmarks(frame, hand_landmarks):
    """Vẽ 21 landmarks và skeleton lên frame OpenCV."""
    coords = landmarks_to_array(hand_landmarks)
    if coords is None:
        return frame

    height, width = frame.shape[:2]
    points = []
    for x, y, _ in coords:
        px = int(np.clip(x * width, 0, width - 1))
        py = int(np.clip(y * height, 0, height - 1))
        points.append((px, py))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)

    return frame


def get_stable_prediction(pred_buffer, smooth_window: int, threshold: float):
    """
    Chỉ trả label khi smooth_window frame gần nhất đều cùng 1 label hợp lệ.
    Đây là smoothing strict để giảm đọc nhầm/audio liên tục.
    """
    if len(pred_buffer) < smooth_window:
        return None, 0.0

    labels = [item[0] for item in pred_buffer]
    confs = [item[1] for item in pred_buffer]

    first_label = labels[0]
    if first_label is None:
        return None, 0.0

    if all(label == first_label for label in labels) and min(confs) >= threshold:
        return first_label, float(np.mean(confs))

    return None, 0.0


def spoken_text(label: str):
    """Chuyển label thành text đọc audio."""
    low = label.lower()
    if low == "space":
        return "space"
    if low in {"delete", "del"}:
        return "delete"
    return label


class AsyncSpeaker:
    """
    Phát âm thanh bằng pyttsx3 trong thread riêng để tránh làm lag webcam.
    Nếu pyttsx3 lỗi, chương trình vẫn chạy bình thường nhưng không đọc audio.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.q = queue.Queue()
        self.thread = None
        self.engine = None

        if not enabled:
            return

        try:
            import pyttsx3

            self.engine = pyttsx3.init()
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            print("[INFO] Audio: pyttsx3 đã bật.")
        except Exception as exc:
            self.enabled = False
            print(f"[WARN] Không khởi tạo được pyttsx3. Tắt audio. Lỗi: {exc}")

    def _worker(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as exc:
                print(f"[WARN] Lỗi khi đọc audio: {exc}")

    def say(self, text: str):
        if self.enabled and text:
            # Không xếp hàng quá nhiều audio nếu máy lag.
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            self.q.put(text)

    def stop(self):
        if self.enabled:
            self.q.put(None)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Realtime ASL static hand sign recognition")
    parser.add_argument("--model", type=str, default="hand_sign_static_mlp.keras", help="Đường dẫn model .keras hoặc .h5")
    parser.add_argument("--label_map", type=str, default="label_map.json", help="Đường dẫn label_map.json")
    parser.add_argument("--hand_landmarker_task", type=str, default=DEFAULT_HAND_LANDMARKER_TASK, help="Đường dẫn hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index, thường là 0 hoặc 1")
    parser.add_argument("--confidence", type=float, default=0.7, help="Ngưỡng confidence để hiển thị/đọc")
    parser.add_argument("--smooth_window", type=int, default=10, help="Số frame liên tiếp cần ổn định")
    parser.add_argument("--speak_cooldown", type=float, default=1.5, help="Thời gian tối thiểu giữa 2 lần đọc audio")
    parser.add_argument("--no_audio", action="store_true", help="Tắt audio pyttsx3")
    return parser


def main():
    args = build_arg_parser().parse_args()

    model_path = Path(args.model)
    label_map_path = Path(args.label_map)
    hand_landmarker_task_path = ensure_hand_landmarker_model(Path(args.hand_landmarker_task))

    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Không tìm thấy label_map.json: {label_map_path}")

    id_to_label = load_label_map(str(label_map_path))
    model = tf.keras.models.load_model(str(model_path))

    print("[INFO] Loaded model:", model_path)
    print("[INFO] Loaded labels:", id_to_label)
    print("[INFO] Loaded HandLandmarker task:", hand_landmarker_task_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Không mở được webcam index {args.camera}. "
            "Thử --camera 1 hoặc kiểm tra quyền camera."
        )

    speaker = AsyncSpeaker(enabled=not args.no_audio)
    pred_buffer = deque(maxlen=args.smooth_window)
    last_spoken_label = None
    last_spoken_time = 0.0

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_landmarker_task_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_index = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Không đọc được frame từ webcam.")
                break

            # Mirror để thao tác tự nhiên hơn khi nhìn webcam.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp phải tăng dần trong VIDEO mode.
            frame_index += 1
            timestamp_ms = int(frame_index * 1000 / 30)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            display_text = "No hand detected"
            display_conf = 0.0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                features = normalize_landmarks(hand_landmarks)
                draw_hand_landmarks(frame, hand_landmarks)

                if features is not None and features.shape == (FEATURE_DIM,):
                    probs = model.predict(features.reshape(1, -1), verbose=0)[0]
                    pred_id = int(np.argmax(probs))
                    conf = float(probs[pred_id])
                    raw_label = id_to_label.get(pred_id, "unknown")

                    if conf >= args.confidence:
                        pred_buffer.append((raw_label, conf))
                    else:
                        pred_buffer.append((None, conf))

                    stable_label, stable_conf = get_stable_prediction(
                        pred_buffer,
                        smooth_window=args.smooth_window,
                        threshold=args.confidence,
                    )

                    if stable_label is not None:
                        display_text = stable_label
                        display_conf = stable_conf

                        now = time.time()
                        should_speak = (
                            stable_label.lower() != "nothing"
                            and stable_label != last_spoken_label
                            and stable_conf >= args.confidence
                            and (now - last_spoken_time) >= args.speak_cooldown
                        )

                        if should_speak:
                            speaker.say(spoken_text(stable_label))
                            last_spoken_label = stable_label
                            last_spoken_time = now
                    else:
                        if conf >= args.confidence:
                            display_text = f"{raw_label} stabilizing"
                            display_conf = conf
                        else:
                            display_text = "Low confidence"
                            display_conf = conf
                else:
                    pred_buffer.clear()
                    display_text = "Invalid landmarks"
            else:
                pred_buffer.clear()
                display_text = "No hand detected"

            if display_conf > 0:
                text = f"{display_text} ({display_conf:.2f})"
            else:
                text = display_text

            cv2.putText(
                frame,
                text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press q to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Static Hand Sign Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    speaker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
