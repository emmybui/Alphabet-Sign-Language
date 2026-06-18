import argparse
import json
import time
import threading
import queue
import urllib.request
from collections import Counter, defaultdict, deque
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

# Màu vẽ cho từng tay.
HAND_COLORS = {
    "Left": (0, 255, 0),
    "Right": (0, 200, 255),
    "Hand": (255, 255, 255),
}


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
    """
    Đọc label_map.json.

    Hỗ trợ các format:
      1. Format mới từ notebook sửa:
         {"label_to_id": {"A": 0}, "id_to_label": {"0": "A"}}
      2. Format id_to_label:
         {"id_to_label": {"0": "A"}}
      3. Format đơn giản:
         {"0": "A"}
      4. Format cũ từ notebook ban đầu:
         {"A": 0}
    """
    with open(label_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "id_to_label" in data:
        id_to_label = {int(k): str(v) for k, v in data["id_to_label"].items()}
    elif "label_to_id" in data:
        id_to_label = {int(v): str(k) for k, v in data["label_to_id"].items()}
    else:
        # {"0": "A"} hoặc {"A": 0}
        first_key = next(iter(data.keys()))
        first_value = data[first_key]
        if str(first_key).isdigit():
            id_to_label = {int(k): str(v) for k, v in data.items()}
        else:
            id_to_label = {int(v): str(k) for k, v in data.items()}

    return dict(sorted(id_to_label.items()))


def landmarks_to_array(hand_landmarks):
    """
    Chuyển output landmarks về np.ndarray shape (21, 3).
    Hỗ trợ MediaPipe Tasks: result.hand_landmarks[i] là list 21 landmarks.
    """
    if hand_landmarks is None:
        return None

    if hasattr(hand_landmarks, "landmark"):
        landmarks = hand_landmarks.landmark
    else:
        landmarks = hand_landmarks

    if len(landmarks) != NUM_LANDMARKS:
        return None

    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalize_landmarks(hand_landmarks, mirror_x: bool = False):
    """
    Normalize landmarks giống lúc training:
      1. Lấy 21 điểm x, y, z.
      2. Trừ tọa độ wrist landmark 0.
      3. Có thể mirror trục x để xử lý tay trái/tay phải.
      4. Scale theo khoảng cách wrist -> middle finger MCP landmark 9.
      5. Flatten thành vector 63 features.
    """
    coords = landmarks_to_array(hand_landmarks)
    if coords is None or coords.shape != (NUM_LANDMARKS, NUM_COORDS):
        return None

    coords = coords - coords[0].copy()

    if mirror_x:
        coords[:, 0] *= -1.0

    scale = np.linalg.norm(coords[MIDDLE_FINGER_MCP_INDEX])
    if scale < EPS:
        return None

    coords = coords / scale
    features = coords.flatten().astype(np.float32)
    if features.shape[0] != FEATURE_DIM:
        return None

    return features


def draw_hand_landmarks(frame, hand_landmarks, hand_label="Hand"):
    """Vẽ 21 landmarks và skeleton lên frame OpenCV, trả về bounding box."""
    coords = landmarks_to_array(hand_landmarks)
    if coords is None:
        return frame, None

    height, width = frame.shape[:2]
    points = []
    for x, y, _ in coords:
        px = int(np.clip(x * width, 0, width - 1))
        py = int(np.clip(y * height, 0, height - 1))
        points.append((px, py))

    color = HAND_COLORS.get(hand_label, HAND_COLORS["Hand"])

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2)

    for point in points:
        cv2.circle(frame, point, 4, color, -1)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 18
    x1 = max(min(xs) - pad, 0)
    y1 = max(min(ys) - pad, 0)
    x2 = min(max(xs) + pad, width - 1)
    y2 = min(max(ys) + pad, height - 1)
    bbox = (x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame, bbox


def get_handedness_label(result, index: int):
    """Lấy label Left/Right từ MediaPipe Tasks nếu có."""
    try:
        handedness_item = result.handedness[index]
        if handedness_item and len(handedness_item) > 0:
            return handedness_item[0].category_name
    except Exception:
        pass
    return f"Hand{index + 1}"


def predict_features(model, features, id_to_label):
    """Predict 1 feature vector, trả về label/confidence/probs."""
    probs = model.predict(features.reshape(1, -1), verbose=0)[0]
    pred_id = int(np.argmax(probs))
    conf = float(probs[pred_id])
    label = id_to_label.get(pred_id, "unknown")
    return label, conf, probs


def predict_best_orientation(model, hand_landmarks, id_to_label, use_mirror_prediction=True):
    """
    Predict với feature gốc và feature mirror_x, chọn orientation có confidence cao hơn.
    Cách này giúp demo bền hơn với tay trái/tay phải hoặc webcam mirror.
    """
    features = normalize_landmarks(hand_landmarks, mirror_x=False)
    if features is None:
        return None, 0.0, None, "invalid"

    label, conf, probs = predict_features(model, features, id_to_label)
    best = (label, conf, probs, "normal")

    if use_mirror_prediction:
        mirrored = normalize_landmarks(hand_landmarks, mirror_x=True)
        if mirrored is not None:
            m_label, m_conf, m_probs = predict_features(model, mirrored, id_to_label)
            if m_conf > conf:
                best = (m_label, m_conf, m_probs, "mirror")

    return best


def get_stable_prediction(pred_buffer, smooth_window: int, threshold: float, majority_ratio: float):
    """
    Majority smoothing cho từng tay.

    Bản cũ yêu cầu toàn bộ frame trong buffer phải giống nhau nên rất khó stable.
    Bản mới lấy nhãn xuất hiện nhiều nhất, nhưng vẫn yêu cầu đủ tỷ lệ majority
    và confidence trung bình vượt threshold.
    """
    if len(pred_buffer) < smooth_window:
        return None, 0.0

    valid_items = [(label, conf) for label, conf in pred_buffer if label is not None and conf >= threshold]
    if not valid_items:
        return None, 0.0

    labels = [item[0] for item in valid_items]
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]

    if top_count / smooth_window < majority_ratio:
        return None, 0.0

    top_confs = [conf for label, conf in valid_items if label == top_label]
    stable_conf = float(np.mean(top_confs))
    if stable_conf < threshold:
        return None, 0.0

    return top_label, stable_conf


def top_k_text(probs, id_to_label, k=3):
    if probs is None:
        return ""
    top_ids = np.argsort(probs)[::-1][:k]
    parts = [f"{id_to_label.get(int(i), 'unknown')}:{float(probs[i]):.2f}" for i in top_ids]
    return " | ".join(parts)


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
    parser = argparse.ArgumentParser(description="Realtime ASL static hand sign recognition — two hands")
    parser.add_argument("--model", type=str, default="hand_sign_static_mlp.keras", help="Đường dẫn model .keras hoặc .h5")
    parser.add_argument("--label_map", type=str, default="label_map.json", help="Đường dẫn label_map.json")
    parser.add_argument("--hand_landmarker_task", type=str, default=DEFAULT_HAND_LANDMARKER_TASK, help="Đường dẫn hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index, thường là 0 hoặc 1")
    parser.add_argument("--num_hands", type=int, default=2, help="Số tay tối đa cần detect/tracking")
    parser.add_argument("--confidence", type=float, default=0.65, help="Ngưỡng confidence để hiển thị/đọc")
    parser.add_argument("--smooth_window", type=int, default=7, help="Số frame dùng để smoothing")
    parser.add_argument("--majority_ratio", type=float, default=0.60, help="Tỷ lệ majority trong smooth_window")
    parser.add_argument("--speak_cooldown", type=float, default=1.5, help="Thời gian tối thiểu giữa 2 lần đọc audio")
    parser.add_argument("--no_audio", action="store_true", help="Tắt audio pyttsx3")
    parser.add_argument("--disable_mirror_prediction", action="store_true", help="Tắt predict thử cả orientation mirror")
    parser.add_argument("--show_top3", action="store_true", help="Hiển thị top-3 prediction cho từng tay")
    parser.add_argument("--min_detection_confidence", type=float, default=0.60, help="MediaPipe detection confidence")
    parser.add_argument("--min_presence_confidence", type=float, default=0.60, help="MediaPipe hand presence confidence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.60, help="MediaPipe tracking confidence")
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
    print("[INFO] Two-hand tracking enabled. num_hands =", args.num_hands)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Không mở được webcam index {args.camera}. "
            "Thử --camera 1 hoặc kiểm tra quyền camera."
        )

    speaker = AsyncSpeaker(enabled=not args.no_audio)

    # Buffer riêng cho từng tay: Left, Right hoặc Hand1/Hand2.
    pred_buffers = defaultdict(lambda: deque(maxlen=args.smooth_window))
    last_spoken_label = {}
    last_spoken_time = defaultdict(lambda: 0.0)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_landmarker_task_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=max(1, args.num_hands),
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
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

            seen_keys = set()
            status_lines = []

            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    hand_label = get_handedness_label(result, idx)
                    hand_key = hand_label if hand_label in {"Left", "Right"} else f"Hand{idx + 1}"
                    seen_keys.add(hand_key)

                    frame, bbox = draw_hand_landmarks(frame, hand_landmarks, hand_label if hand_label in HAND_COLORS else "Hand")

                    label, conf, probs, orientation = predict_best_orientation(
                        model,
                        hand_landmarks,
                        id_to_label,
                        use_mirror_prediction=not args.disable_mirror_prediction,
                    )

                    if label is not None and conf >= args.confidence:
                        pred_buffers[hand_key].append((label, conf))
                    else:
                        pred_buffers[hand_key].append((None, conf))

                    stable_label, stable_conf = get_stable_prediction(
                        pred_buffers[hand_key],
                        smooth_window=args.smooth_window,
                        threshold=args.confidence,
                        majority_ratio=args.majority_ratio,
                    )

                    if stable_label is not None:
                        display_text = f"{hand_key}: {stable_label} ({stable_conf:.2f})"
                        now = time.time()
                        should_speak = (
                            stable_label.lower() != "nothing"
                            and stable_label != last_spoken_label.get(hand_key)
                            and (now - last_spoken_time[hand_key]) >= args.speak_cooldown
                        )
                        if should_speak:
                            speaker.say(spoken_text(stable_label))
                            last_spoken_label[hand_key] = stable_label
                            last_spoken_time[hand_key] = now
                    else:
                        if label is not None and conf >= args.confidence:
                            display_text = f"{hand_key}: {label} stabilizing ({conf:.2f})"
                        elif label is not None:
                            display_text = f"{hand_key}: low confidence {label} ({conf:.2f})"
                        else:
                            display_text = f"{hand_key}: invalid landmarks"

                    if args.show_top3 and probs is not None:
                        display_text += f" | {orientation} | {top_k_text(probs, id_to_label, k=3)}"

                    status_lines.append(display_text)

                    if bbox is not None:
                        x1, y1, _, _ = bbox
                        color = HAND_COLORS.get(hand_label, HAND_COLORS["Hand"])
                        cv2.putText(
                            frame,
                            display_text[:55],
                            (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
            else:
                status_lines.append("No hand detected")

            # Clear buffer cho tay đã biến mất khỏi frame.
            for key in list(pred_buffers.keys()):
                if key not in seen_keys:
                    pred_buffers[key].clear()

            # Status panel góc trái.
            y0 = 35
            for line in status_lines[:4]:
                cv2.putText(
                    frame,
                    line[:95],
                    (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y0 += 28

            cv2.putText(
                frame,
                "Press q to quit | Tracking up to 2 hands",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("ASL Static Hand Sign Demo — Two Hands", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    speaker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
