import cv2
import json
from camera_motion import CameraMotionEstimator, estimate_point_displacement
from yolo_tracker import ObjectMotionTracker
from motion_plots import plot_motion_results

VIDEO_PATH = "./test_video.mp4"
OUTPUT_VIDEO_PATH = "./output_tracked.avi"
RESULT_FILE = "./object_motion_camera_compensated.json"

PX_THRESHOLD = 5.0
AREA_RATIO_THRESHOLD = 0.08

# номінальна тривалість камерного інтервалу; dt для об'єктів беремо
# окремо з obj["dt_sec"], бо він може відхилятись від цих 0.5с
CAM_INTERVAL_SEC = 0.5

def describe_motion(dx, dy, area_ratio):
    directions = []

    if dx > PX_THRESHOLD:
        directions.append(f"Right (+{dx:.1f}px)")
    elif dx < -PX_THRESHOLD:
        directions.append(f"Left ({dx:.1f}px)")

    if dy > PX_THRESHOLD:
        directions.append(f"Down (+{dy:.1f}px)")
    elif dy < -PX_THRESHOLD:
        directions.append(f"Up ({dy:.1f}px)")

    if area_ratio > 1 + AREA_RATIO_THRESHOLD:
        directions.append(f"Forward / Approaching (+{(area_ratio - 1) * 100:.1f}%)")
    elif area_ratio < 1 - AREA_RATIO_THRESHOLD:
        directions.append(f"Backward / Moving Away (-{(1 - area_ratio) * 100:.1f}%)")

    return ", ".join(directions) if directions else "Stationary / In Place"

def describe_camera(dx, dy, scale):
    directions = []

    if dx > PX_THRESHOLD:
        directions.append(f"Right (+{dx:.1f}px)")
    elif dx < -PX_THRESHOLD:
        directions.append(f"Left ({dx:.1f}px)")

    if dy > PX_THRESHOLD:
        directions.append(f"Down (+{dy:.1f}px)")
    elif dy < -PX_THRESHOLD:
        directions.append(f"Up ({dy:.1f}px)")

    if scale > 1 + AREA_RATIO_THRESHOLD:
        directions.append(f"Zoom+ (+{(scale - 1) * 100:.1f}%)")
    elif scale < 1 - AREA_RATIO_THRESHOLD:
        directions.append(f"Zoom- (-{(1 - scale) * 100:.1f}%)")

    return ", ".join(directions) if directions else "Camera is stationary"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: failed to open video {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_sec_frames = max(1, int(round(fps / 2.0)))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print(f"Processing video: {width}x{height} @ {fps:.2f} FPS")

    cam_estimator = CameraMotionEstimator(width, height, fps)
    obj_tracker = ObjectMotionTracker()

    result_log = []

    ret, frame = cap.read()
    if not ret:
        print("Error: failed to read video")
        return

    # спершу трекер об'єктів (щоб мати свіжі bbox для маски ORB), і тільки
    # потім оцінювач руху камери — інакше RANSAC не відрізнить фон від
    # великого рухомого об'єкта в кадрі
    obj_tracker.process_frame(frame, frame_count=1, current_time=0.0)
    exclude_bboxes = [info["bbox"] for info in obj_tracker.active_tracks.values()]
    cam_estimator.init_first_frame(frame, exclude_bboxes=exclude_bboxes)

    active_ttc_dict = {}
    out.write(obj_tracker.draw(frame, active_ttc_dict))

    frame_count = 1

    # єдиний лічильник кадрів для обох інтервалів (об'єктного і камерного),
    # щоб вони закривались синхронно — інакше TTC-підпис блиматиме на межі
    # кожного інтервалу
    period_frame_count = 0
    interval_number = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = (frame_count - 1) / fps

        obj_tracker.process_frame(frame, frame_count, current_time)
        exclude_bboxes = [info["bbox"] for info in obj_tracker.active_tracks.values()]
        cam_estimator.process_frame(frame, exclude_bboxes=exclude_bboxes)

        period_frame_count += 1

        if period_frame_count >= half_sec_frames:
            t_start = (interval_number - 1) * 0.5
            t_end = t_start + 0.5

            obj_stats_list = obj_tracker.get_interval_motion(frame_count, half_sec_frames)

            cam_stats = cam_estimator.get_interval_stats()
            cam_dx_total = cam_stats["dx_px_per_sec"] * CAM_INTERVAL_SEC
            cam_dy_total = cam_stats["dy_px_per_sec"] * CAM_INTERVAL_SEC
            cam_scale = cam_stats["scale"] if cam_stats["scale"] > 0 else 1.0
            cam_dir_str = describe_camera(cam_dx_total, cam_dy_total, cam_scale)

            print(f"\n--- INTERVAL {interval_number} ({t_start:.1f}s - {t_end:.1f}s) ---")
            print(f"  X Speed:     {cam_stats['dx_px_per_sec']:+.2f} px/s")
            print(f"  Y Speed:     {cam_stats['dy_px_per_sec']:+.2f} px/s")
            print(f"  Scale:       {cam_scale:.4f}")
            print(f"  Valid steps: {cam_stats['valid_steps']}")

            interval_record = {
                "interval": interval_number,
                "t_start": round(t_start, 3),
                "t_end": round(t_end, 3),
                "camera_motion": {
                    "dx_px": round(cam_dx_total, 2),
                    "dy_px": round(cam_dy_total, 2),
                    "dx_px_per_sec": round(cam_stats["dx_px_per_sec"], 2),
                    "dy_px_per_sec": round(cam_stats["dy_px_per_sec"], 2),
                    "scale": round(cam_scale, 4),
                    "valid_steps": cam_stats["valid_steps"],
                    "description": cam_dir_str,
                },
                "objects": [],
            }

            if not obj_stats_list:
                print("  > No active objects for analysis.")

            active_ttc_dict = {}

            for obj in obj_stats_list:
                obj_dt = obj["dt_sec"]  # реальний dt цього об'єкта, а не завжди 0.5с

                current_box = obj_tracker.active_tracks.get(obj["track_id"], {}).get("bbox")
                if current_box:
                    x1, y1, x2, y2 = current_box
                    anchor_x = (x1 + x2) / 2.0
                    anchor_y = (y1 + y2) / 2.0
                else:
                    # об'єкт щойно зник з треку — компенсуємо як для центру кадру
                    anchor_x, anchor_y = width / 2.0, height / 2.0

                # зсув камери саме в позиції об'єкта, масштабований під реальний dt
                dt_scale = obj_dt / CAM_INTERVAL_SEC if CAM_INTERVAL_SEC > 0 else 1.0
                cam_dx_at_obj, cam_dy_at_obj = estimate_point_displacement(cam_stats, anchor_x, anchor_y)
                cam_dx_at_obj *= dt_scale
                cam_dy_at_obj *= dt_scale
                cam_scale_for_obj = cam_scale ** dt_scale if cam_scale > 0 else 1.0

                real_dx = obj["dx_px"] - cam_dx_at_obj
                real_dy = obj["dy_px"] - cam_dy_at_obj
                real_area_ratio = obj["area_ratio"] / (cam_scale_for_obj ** 2)

                screen_dir_str = describe_motion(obj["dx_px"], obj["dy_px"], obj["area_ratio"])
                real_dir_str = describe_motion(real_dx, real_dy, real_area_ratio)

                # TTC за гіперболічною моделлю: при сталій швидкості зближення
                # площа росте як 1/(1-t/T)^2, звідки T = 2 / (темп росту площі).
                # Залежить лише від темпу росту площі, а не від того, скільки
                # ще кадру лишилось вільним
                time_to_collision = None
                if real_area_ratio > 1.0 and obj_dt > 0:
                    area_growth_rate = (real_area_ratio - 1.0) / obj_dt
                    if area_growth_rate > 0:
                        time_to_collision = 2.0 / area_growth_rate

                if time_to_collision is not None:
                    active_ttc_dict[obj["track_id"]] = time_to_collision

                print(f"  > [{obj['class_name']} {obj['track_id']}]")
                print(f"      on screen (with camera movement):     {screen_dir_str}")
                print(f"      in reality (camera stationary):     {real_dir_str}")
                if time_to_collision is not None:
                    print(f"      time to collision (TTC):     {time_to_collision:.2f} sec")
                else:
                    print(f"      time to collision (TTC):     N/A (not approaching)")

                interval_record["objects"].append({
                    "track_id": obj["track_id"],
                    "class_name": obj["class_name"],
                    "dt_sec": round(obj_dt, 3),
                    "screen_motion": {
                        "dx_px": round(obj["dx_px"], 2),
                        "dy_px": round(obj["dy_px"], 2),
                        "area_ratio": round(obj["area_ratio"], 4),
                        "description": screen_dir_str,
                    },
                    "real_motion": {
                        "dx_px": round(real_dx, 2),
                        "dy_px": round(real_dy, 2),
                        "area_ratio": round(real_area_ratio, 4),
                        "description": real_dir_str,
                    },
                    "time_to_collision_sec": float(round(time_to_collision, 2)) if time_to_collision is not None else None,
                })

            result_log.append(interval_record)

            period_frame_count = 0
            interval_number += 1

        out.write(obj_tracker.draw(frame, active_ttc_dict))

    cap.release()
    out.release()

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result_log, f, ensure_ascii=False, indent=2)

    print(f"  Video with annotations: {OUTPUT_VIDEO_PATH}")
    print(f"  Camera + objects (screen and compensated): {RESULT_FILE}")

    plot_motion_results(json_path=RESULT_FILE, output_dir="./plots")

if __name__ == "__main__":
    main()