import cv2
import numpy as np

def estimate_point_displacement(stats, x, y):
    # Зсув від руху камери в довільній точці кадру (x, y), а не тільки в центрі.

    # Афінна модель: p' = [[a,-b],[b,a]] @ p + [tx,ty]
    # Звідси d(p) = (a-1)*x - b*y + tx, b*x + (a-1)*y + ty — лінійно залежить
    # від (x, y), тож досить сум a-1/b/tx/ty за інтервал (get_interval_stats)
    # щоб підставити будь-яку точку.

    # При чистій трансляції (a=1, b=0) зводиться просто до (tx, ty) для будь-якої
    # точки. При зумі результат вже залежить від відстані до центру кадру.
    
    dx = stats["sum_a_minus1"] * x - stats["sum_b"] * y + stats["sum_tx"]
    dy = stats["sum_b"] * x + stats["sum_a_minus1"] * y + stats["sum_ty"]
    return dx, dy

class CameraMotionEstimator:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.center_x = width / 2.0
        self.center_y = height / 2.0
        self.fps = fps

        self.orb = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        # зсув у центрі кадру — для зведеного camera_motion в JSON і графіках
        self.period_dx = []
        self.period_dy = []
        self.period_scales = []

        # сирі компоненти афінного перетворення по кожному кадру — потрібні,
        # щоб рахувати зсув камери в довільній точці (для компенсації руху
        # конкретного об'єкта), а не тільки в центрі
        self.period_a_minus1 = []
        self.period_b = []
        self.period_tx = []
        self.period_ty = []

        self._initialized = False

    def _build_object_mask(self, shape, exclude_bboxes):
        # Маска для ORB: 0 в прямокутниках трекнутих об'єктів, 255 — решта.

        # Без цього RANSAC бере keypoints звідусіль, включно з великим рухомим
        # об'єктом (машина/людина близько до камери може займати чималу частку
        # кадру) — і за достатньої кількості інлайєрів на ньому цілком реально
        # визнає ЙОГО рух домінантним і підставить як рух камери.
        
        if not exclude_bboxes:
            return None

        mask = np.full(shape, 255, dtype=np.uint8)
        h, w = shape
        for bbox in exclude_bboxes:
            x1, y1, x2, y2 = bbox
            x1 = int(max(0, min(w, x1)))
            x2 = int(max(0, min(w, x2)))
            y1 = int(max(0, min(h, y1)))
            y2 = int(max(0, min(h, y2)))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    def init_first_frame(self, frame, exclude_bboxes=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._build_object_mask(gray.shape, exclude_bboxes)
        self.prev_gray = gray
        self.prev_keypoints, self.prev_descriptors = self.orb.detectAndCompute(gray, mask)
        self._initialized = True

    def process_frame(self, frame, exclude_bboxes=None):
        if not self._initialized:
            self.init_first_frame(frame, exclude_bboxes)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._build_object_mask(gray.shape, exclude_bboxes)
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)

        if self.prev_descriptors is not None and descriptors is not None:
            matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= 6:
                pts_prev = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
                pts_curr = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

                M, inliers = cv2.estimateAffinePartial2D(
                    pts_prev,
                    pts_curr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0,
                    maxIters=2000,
                )

                if M is not None and inliers is not None and np.sum(inliers) >= 5:
                    a = M[0, 0]
                    b = M[1, 0]
                    tx = M[0, 2]
                    ty = M[1, 2]

                    scale = np.sqrt(a**2 + b**2)

                    if 0.5 <= scale <= 2.0:
                        dx_center = (a - 1.0) * self.center_x - b * self.center_y + tx
                        dy_center = b * self.center_x + (a - 1.0) * self.center_y + ty

                        if abs(dx_center) < self.width * 0.5 and abs(dy_center) < self.height * 0.5:
                            self.period_dx.append(dx_center)
                            self.period_dy.append(dy_center)
                            self.period_scales.append(scale)

                            self.period_a_minus1.append(a - 1.0)
                            self.period_b.append(b)
                            self.period_tx.append(tx)
                            self.period_ty.append(ty)

        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def get_interval_stats(self):
        if len(self.period_dx) > 0:
            total_dx_px = np.sum(self.period_dx)
            total_dy_px = np.sum(self.period_dy)

            actual_duration = len(self.period_dx) / self.fps
            dx_sec = total_dx_px / actual_duration
            dy_sec = total_dy_px / actual_duration

            total_scale = float(np.prod(self.period_scales))
            valid_steps = len(self.period_dx)

            sum_a_minus1 = float(np.sum(self.period_a_minus1))
            sum_b = float(np.sum(self.period_b))
            sum_tx = float(np.sum(self.period_tx))
            sum_ty = float(np.sum(self.period_ty))
        else:
            dx_sec = 0.0
            dy_sec = 0.0
            total_scale = 1.0
            valid_steps = 0

            sum_a_minus1 = 0.0
            sum_b = 0.0
            sum_tx = 0.0
            sum_ty = 0.0

        self.period_dx.clear()
        self.period_dy.clear()
        self.period_scales.clear()
        self.period_a_minus1.clear()
        self.period_b.clear()
        self.period_tx.clear()
        self.period_ty.clear()

        return {
            "dx_px_per_sec": float(dx_sec),
            "dy_px_per_sec": float(dy_sec),
            "scale": total_scale,
            "valid_steps": valid_steps,

            # для estimate_point_displacement() — зсув камери в позиції будь-якого об'єкта
            "sum_a_minus1": sum_a_minus1,
            "sum_b": sum_b,
            "sum_tx": sum_tx,
            "sum_ty": sum_ty,
        }