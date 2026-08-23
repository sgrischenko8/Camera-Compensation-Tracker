import cv2
import numpy as np
 
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

        self.period_dx = []
        self.period_dy = []
        self.period_scales = []

        self._initialized = False

    def init_first_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.prev_keypoints, self.prev_descriptors = self.orb.detectAndCompute(gray, None)
        self._initialized = True

    def process_frame(self, frame):
        if not self._initialized:
            self.init_first_frame(frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

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
        else:
            dx_sec = 0.0
            dy_sec = 0.0
            total_scale = 1.0
            valid_steps = 0

        self.period_dx.clear()
        self.period_dy.clear()
        self.period_scales.clear()

        return {
            "dx_px_per_sec": float(dx_sec),
            "dy_px_per_sec": float(dy_sec),
            "scale": total_scale,
            "valid_steps": valid_steps,
        }
