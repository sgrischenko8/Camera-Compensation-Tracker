import cv2
import json
from camera_motion import CameraMotionEstimator
from yolo_tracker import ObjectMotionTracker
from motion_plots import plot_motion_results

VIDEO_PATH = "./test_video.mp4"
OUTPUT_VIDEO_PATH = "./output_tracked.avi"
RESULT_FILE = "./object_motion_camera_compensated.json"

PX_THRESHOLD = 5.0
AREA_RATIO_THRESHOLD = 0.08

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
    frame_area = width * height
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

    cam_estimator.init_first_frame(frame)
    obj_tracker.process_frame(frame, frame_count=1, current_time=0.0)
    active_ttc_dict = {}
    out.write(obj_tracker.draw(frame, active_ttc_dict))

    frame_count = 1           
    cam_period_frame_count = 0   
    cam_interval_number = 1

    pending_obj_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = (frame_count - 1) / fps

        cam_estimator.process_frame(frame)
        cam_period_frame_count += 1

        obj_tracker.process_frame(frame, frame_count, current_time)

        if frame_count % half_sec_frames == 0:
            obj_t_start = max(0.0, current_time - 0.5)
            obj_stats_list = obj_tracker.get_interval_motion(frame_count, half_sec_frames)
            pending_obj_result = {
                "t_start": obj_t_start,
                "t_end": current_time,
                "frame": frame_count,
                "objects": obj_stats_list,
            }
            active_ttc_dict.clear()
            for obj in obj_stats_list:
                pass
        
        if cam_period_frame_count >= half_sec_frames:
            cam_t_start = (cam_interval_number - 1) * 0.5
            cam_t_end = cam_t_start + 0.5

            cam_stats = cam_estimator.get_interval_stats()
            cam_dx_total = cam_stats["dx_px_per_sec"] * 0.5
            cam_dy_total = cam_stats["dy_px_per_sec"] * 0.5
            cam_scale = cam_stats["scale"] if cam_stats["scale"] > 0 else 1.0
            cam_dir_str = describe_camera(cam_dx_total, cam_dy_total, cam_scale)

            print(f"\n--- INTERVAL {cam_interval_number} ({cam_t_start:.1f}s - {cam_t_end:.1f}s) ---")
            print(f"  X Speed:     {cam_stats['dx_px_per_sec']:+.2f} px/s")
            print(f"  Y Speed:     {cam_stats['dy_px_per_sec']:+.2f} px/s")
            print(f"  Scale:       {cam_scale:.4f}")
            print(f"  Valid steps: {cam_stats['valid_steps']}")

            interval_record = {
                "interval": cam_interval_number,
                "t_start": round(cam_t_start, 3),
                "t_end": round(cam_t_end, 3),
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

            obj_stats_list = pending_obj_result["objects"] if pending_obj_result else []
            if pending_obj_result is None:
                print("  [!] No object tracker data for this interval (out of sync)")

            if not obj_stats_list:
                print("  > No active objects for analysis.")

            active_ttc_dict = {}

            for obj in obj_stats_list:
                real_dx = obj["dx_px"] - cam_dx_total
                real_dy = obj["dy_px"] - cam_dy_total
                real_area_ratio = obj["area_ratio"] / (cam_scale ** 2)

                screen_dir_str = describe_motion(obj["dx_px"], obj["dy_px"], obj["area_ratio"])
                real_dir_str = describe_motion(real_dx, real_dy, real_area_ratio)
                
                # --- (Time-to-Collision, TTC) ---
                time_to_collision = None
                
                if real_area_ratio > 1.0:
                    current_box = obj_tracker.active_tracks.get(obj["track_id"], {}).get("bbox")
                    if current_box:
                        x1, y1, x2, y2 = current_box
                        current_area = max(1.0, (x2 - x1) * (y2 - y1))
                        
                        area_growth_per_sec = (real_area_ratio - 1.0) / 0.5
                        
                        if area_growth_per_sec > 0:
                            remaining_area = frame_area - current_area
                            if remaining_area > 0:
                                time_to_collision = remaining_area / (current_area * area_growth_per_sec)
                            else:
                                time_to_collision = 0.0

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

            pending_obj_result = None
            cam_period_frame_count = 0
            cam_interval_number += 1

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