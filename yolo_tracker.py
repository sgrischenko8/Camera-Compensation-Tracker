import cv2
from collections import defaultdict, deque, Counter
from ultralytics import YOLO

class ObjectMotionTracker:
    def __init__(self, model_name="yolov8n.pt", detection_interval=3, conf_threshold=0.35,
                 max_history_seconds=2.0):
        self.model = YOLO(model_name)
        self.detection_interval = detection_interval
        self.conf_threshold = conf_threshold
        self.max_history_seconds = max_history_seconds  # скільки історії тримати на трек

        self.track_history = defaultdict(deque)
        self.active_tracks = {}

        # голосування за клас: одна невдала детекція на межовому кадрі
        # більше не перефарбовує весь track_id в інший клас
        self.class_votes = defaultdict(Counter)
        self.class_names_map = {}

    def process_frame(self, frame, frame_count, current_time):
        is_detection_frame = (frame_count - 1) % self.detection_interval == 0

        if is_detection_frame:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            current_active = {}
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                clss = results[0].boxes.cls.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                    # приводимо до звичайного float, бо numpy.float32 потім
                    # падає на json.dump ("Object of type float32 is not JSON serializable")
                    x1, y1, x2, y2 = (float(v) for v in box)
                    detected_name = self.model.names[cls_id]

                    self.class_votes[track_id][detected_name] += 1
                    resolved_name = self.class_votes[track_id].most_common(1)[0][0]
                    self.class_names_map[track_id] = resolved_name

                    current_active[track_id] = {
                        "bbox": [x1, y1, x2, y2],
                        "class_name": resolved_name,
                        "conf": float(conf),
                    }
            self.active_tracks = current_active

        # історію пишемо щокадру (для плавної відмальовки), але позначаємо
        # is_detection — get_interval_motion бере тільки реальні детекції,
        # щоб не рахувати dx/dt по дубльованих позиціях
        for track_id, info in self.active_tracks.items():
            x1, y1, x2, y2 = info["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = (x2 - x1) * (y2 - y1)

            self.track_history[track_id].append({
                "frame": frame_count,
                "time": current_time,
                "cx": cx,
                "cy": cy,
                "area": area,
                "is_detection": is_detection_frame,
            })

        self._trim_history(current_time)

    def _trim_history(self, current_time):
        # чистимо старі точки і повністю забуваємо треки, яких давно не видно
        cutoff = current_time - self.max_history_seconds
        dead_tracks = []

        for track_id, history in self.track_history.items():
            while history and history[0]["time"] < cutoff:
                history.popleft()
            if not history:
                dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self.track_history[track_id]
            self.class_votes.pop(track_id, None)
            self.class_names_map.pop(track_id, None)

    def get_interval_motion(self, frame_count, half_sec_frames):
        result = []
        for track_id, history in self.track_history.items():
            # беремо тільки точки з реальних детекцій у межах вікна
            recent = [
                h for h in history
                if frame_count - half_sec_frames <= h["frame"] <= frame_count
                and h["is_detection"]
            ]

            if len(recent) >= 2:
                first = recent[0]
                last = recent[-1]

                dt = last["time"] - first["time"]
                if dt <= 0:
                    continue

                dx = last["cx"] - first["cx"]
                dy = last["cy"] - first["cy"]
                area_ratio = last["area"] / first["area"] if first["area"] > 0 else 1.0

                c_name = self.class_names_map.get(track_id, "object")
                result.append({
                    "track_id": int(track_id),
                    "class_name": c_name,
                    "dx_px": float(dx),
                    "dy_px": float(dy),
                    "area_ratio": float(area_ratio),
                    "dt_sec": float(dt),  # реальний dt, а не завжди рівно 0.5с
                })
        return result

    def draw(self, frame, ttc_dict=None):
        for track_id, info in self.active_tracks.items():
            x1, y1, x2, y2 = map(int, info["bbox"])
            c_name = info["class_name"]
            conf = info["conf"]

            ttc = ttc_dict.get(track_id)
            if ttc is not None:
                label = f"{c_name} {track_id} | ttc: {ttc:.1f}s"
            else:
                label = f"{c_name} {track_id} ({int(conf * 100)}%)"

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame