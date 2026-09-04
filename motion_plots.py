import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# зелений — рух "в плюс" (праворуч/вниз), червоний — "в мінус" (ліворуч/вгору)
DIR_POSITIVE_COLOR = "tab:green"
DIR_NEGATIVE_COLOR = "tab:red"

def _direction_colors(deltas):
    return [DIR_POSITIVE_COLOR if d >= 0 else DIR_NEGATIVE_COLOR for d in deltas]

def _direction_legend_handles(positive_label, negative_label):
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DIR_POSITIVE_COLOR,
               markeredgecolor="black", markersize=8, label=positive_label),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DIR_NEGATIVE_COLOR,
               markeredgecolor="black", markersize=8, label=negative_label),
    ]

# все, що більше цього значення, на графіку просто обрізається
TTC_Y_LIMIT = 1000

def _load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _time_axis(data):
    return [interval["t_end"] for interval in data]

def _collect_object_series(data, motion_key):
    series = {}

    for interval in data:
        t = interval["t_end"]
        for obj in interval["objects"]:
            tid = obj["track_id"]
            if tid not in series:
                series[tid] = {
                    "t": [],
                    "dx": [],
                    "dy": [],
                    "class_name": obj.get("class_name", "об'єкт"),
                }
            series[tid]["t"].append(t)
            series[tid]["dx"].append(obj[motion_key]["dx_px"])
            series[tid]["dy"].append(obj[motion_key]["dy_px"])

    return series

def _collect_ttc_series(data):
    series = {}

    for interval in data:
        t = interval["t_end"]
        for obj in interval["objects"]:
            ttc = obj.get("time_to_collision_sec")
            if ttc is None:
                continue
            tid = obj["track_id"]
            if tid not in series:
                series[tid] = {"t": [], "ttc": [], "class_name": obj.get("class_name", "об'єкт")}
            series[tid]["t"].append(t)
            series[tid]["ttc"].append(ttc)

    return series

def plot_camera_motion(data, output_path):
    t = _time_axis(data)
    # dx_px/dy_px в JSON — це приріст за інтервал (0.5с), тому сумуємо накопичувально
    dx_delta = [interval["camera_motion"]["dx_px"] for interval in data]
    dy_delta = [interval["camera_motion"]["dy_px"] for interval in data]
    dx_cum = np.cumsum(dx_delta)
    dy_cum = np.cumsum(dy_delta)

    fig, (ax_dx, ax_dy) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Переміщення камери (накопичене зміщення від старту)")

    ax_dx.plot(t, dx_cum, color="tab:blue", linewidth=1.3, zorder=1)
    ax_dx.scatter(t, dx_cum, c=_direction_colors(dx_delta), edgecolors="black",
                  linewidths=0.5, s=45, zorder=2)
    ax_dx.axhline(0, color="gray", linewidth=0.8)
    ax_dx.set_ylabel("переміщення X, пікс")
    ax_dx.set_title("Накопичене переміщення камери по X")
    ax_dx.grid(True, alpha=0.3)
    ax_dx.legend(handles=_direction_legend_handles("рух вправо", "рух вліво"),
                 fontsize=8, loc="best")

    ax_dy.plot(t, dy_cum, color="tab:orange", linewidth=1.3, zorder=1)
    ax_dy.scatter(t, dy_cum, c=_direction_colors(dy_delta), edgecolors="black",
                  linewidths=0.5, s=45, zorder=2)
    ax_dy.axhline(0, color="gray", linewidth=0.8)
    ax_dy.set_ylabel("переміщення Y, пікс")
    ax_dy.set_xlabel("час, с")
    ax_dy.set_title("Накопичене переміщення камери по Y")
    ax_dy.grid(True, alpha=0.3)
    ax_dy.legend(handles=_direction_legend_handles("рух вниз", "рух вгору"),
                 fontsize=8, loc="best")
    # у піксельних координатах Y росте вниз, тож інвертуємо вісь, щоб графік не був дзеркальним
    ax_dy.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_object_real_motion(data, output_path):
    # тут беремо real_motion — рух об'єкта з уже вирахуваним рухом камери
    series = _collect_object_series(data, "real_motion")

    fig, (ax_dx, ax_dy) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("Власне переміщення об'єктів (з компенсацією руху камери)")

    track_colors = plt.cm.tab10.colors

    for idx, (tid, s) in enumerate(series.items()):
        color = track_colors[idx % len(track_colors)]
        label = f"{s['class_name']} #{tid}"
        dx_cum = np.cumsum(s["dx"])
        dy_cum = np.cumsum(s["dy"])

        ax_dx.plot(s["t"], dx_cum, color=color, linewidth=1.3, label=label, zorder=1)
        ax_dx.scatter(s["t"], dx_cum, c=_direction_colors(s["dx"]), edgecolors=color,
                      linewidths=1.2, s=40, zorder=2)

        ax_dy.plot(s["t"], dy_cum, color=color, linewidth=1.3, label=label, zorder=1)
        ax_dy.scatter(s["t"], dy_cum, c=_direction_colors(s["dy"]), edgecolors=color,
                      linewidths=1.2, s=40, zorder=2)

    ax_dx.axhline(0, color="gray", linewidth=0.8)
    ax_dx.set_ylabel("переміщення X, пікс")
    ax_dx.set_title("Накопичене реальне переміщення об'єктів по X")
    ax_dx.grid(True, alpha=0.3)
    objects_legend_dx = ax_dx.legend(fontsize=7, loc="upper left", title="Об'єкти")
    ax_dx.add_artist(objects_legend_dx)
    ax_dx.legend(handles=_direction_legend_handles("рух вправо", "рух вліво"),
                 fontsize=7, loc="lower right", title="Напрямок")

    ax_dy.axhline(0, color="gray", linewidth=0.8)
    ax_dy.set_ylabel("переміщення Y, пікс")
    ax_dy.set_xlabel("час, с")
    ax_dy.set_title("Накопичене реальне переміщення об'єктів по Y")
    ax_dy.grid(True, alpha=0.3)
    objects_legend_dy = ax_dy.legend(fontsize=7, loc="upper left", title="Об'єкти")
    ax_dy.add_artist(objects_legend_dy)
    ax_dy.legend(handles=_direction_legend_handles("рух вниз", "рух вгору"),
                 fontsize=7, loc="lower right", title="Напрямок")
    ax_dy.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_time_to_collision(data, output_path):
    series = _collect_ttc_series(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Час до зіткнення (TTC)")

    if not series:
        ax.text(0.5, 0.5, "Немає даних про зближення (TTC)", ha="center", va="center")
    else:
        for tid, s in series.items():
            label = f"{s['class_name']} #{tid}"
            ax.plot(s["t"], s["ttc"], marker="o", label=label)

        # Верхня межа адаптується до даних, але ніколи не перевищує 1000 с.
        # Значення TTC понад межу залишаються в даних, але не потрапляють
        # у видиму область графіка.
        max_ttc = max(max(s["ttc"]) for s in series.values() if s["ttc"])
        y_max = min(TTC_Y_LIMIT, max_ttc)
        if y_max <= 0:
            y_max = TTC_Y_LIMIT
        ax.set_ylim(0, y_max)

    ax.set_xlabel("час, с")
    ax.set_ylabel(f"TTC, с (показано до {TTC_Y_LIMIT} с)")
    ax.grid(True, alpha=0.3)
    if series:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_motion_results(json_path="./object_motion_camera_compensated.json",
                         output_dir="./plots"):
    # Будує 3 графіки з JSON-результатів: рух камери, реальний рух об'єктів, TTC.

    os.makedirs(output_dir, exist_ok=True)
    data = _load_data(json_path)

    paths = {
        "camera_motion": os.path.join(output_dir, "camera_motion.png"),
        "object_real_motion": os.path.join(output_dir, "object_real_motion.png"),
        "time_to_collision": os.path.join(output_dir, "time_to_collision.png"),
    }

    plot_camera_motion(data, paths["camera_motion"])
    plot_object_real_motion(data, paths["object_real_motion"])
    plot_time_to_collision(data, paths["time_to_collision"])

    print("Графіки збережено:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    return paths

if __name__ == "__main__":
    plot_motion_results()