#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class PandaCameraViewer:
    def __init__(self):
        rospy.init_node('panda_camera_viewer', anonymous=True)

        # Ensure this is the exact topic you echoed as rgb8
        self.image_topic = "/robot2/camera/image_raw"

        self.bridge = CvBridge()
        self.frame = None
        self.frame_ts = 0.0

        # Tracker state
        self.tracker = None
        self.tracker_bbox = None
        self.tracker_ok = False
        self.last_detect_time = 0.0

        # Default green HSV (tightened for fewer false positives).
        # OpenCV HSV: H[0..179], S[0..255], V[0..255][6][12]
        self.h_lower, self.s_lower, self.v_lower = 45, 100, 80
        self.h_upper, self.s_upper, self.v_upper = 85, 255, 255

        # Morphology and area thresholds
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.min_area = 150      # allow small objects
        self.max_area = 20000    # cap to avoid big background grabs
        self.redetect_interval = 0.4
        self.merge_history = 5
        self.bbox_history = []

        # Shape-quality thresholds for small-object robustness
        self.min_solidity = 0.85  # compact shape favored[19][7]
        self.min_extent = 0.45    # fill inside bbox
        self.ar_min, self.ar_max = 0.4, 2.5  # acceptable aspect ratio

        # Visualization zoom config
        self.zoom_factor = 1.7  # 1.0 = no zoom; >1 zooms in
        self.min_crop_size_px = 64  # avoid too small ROI at high zoom

        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.loginfo("Panda camera live stream with green-object tracking + zoom started. Press 'q' to quit.")

        # Optional: live HSV tuning trackbars
        self.enable_hsv_trackbars = True
        if self.enable_hsv_trackbars:
            cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("H_low", "HSV Tuner", self.h_lower, 179, lambda v: None)
            cv2.createTrackbar("S_low", "HSV Tuner", self.s_lower, 255, lambda v: None)
            cv2.createTrackbar("V_low", "HSV Tuner", self.v_lower, 255, lambda v: None)
            cv2.createTrackbar("H_high", "HSV Tuner", self.h_upper, 179, lambda v: None)
            cv2.createTrackbar("S_high", "HSV Tuner", self.s_upper, 255, lambda v: None)
            cv2.createTrackbar("V_high", "HSV Tuner", self.v_upper, 255, lambda v: None)

    def image_callback(self, msg):
        try:
            # msg is rgb8; request bgr8 for OpenCV processing[4]
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.frame = frame
            self.frame_ts = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else time.time()
            rospy.loginfo_once("Image encoding confirmed: rgb8 -> bgr8 via CvBridge")
        except Exception as e:
            rospy.logerr("CV Bridge error: %s", e)

    def _maybe_update_hsv_from_trackbars(self):
        if not self.enable_hsv_trackbars:
            return
        self.h_lower = cv2.getTrackbarPos("H_low", "HSV Tuner")
        self.s_lower = cv2.getTrackbarPos("S_low", "HSV Tuner")
        self.v_lower = cv2.getTrackbarPos("V_low", "HSV Tuner")
        self.h_upper = cv2.getTrackbarPos("H_high", "HSV Tuner")
        self.s_upper = cv2.getTrackbarPos("S_high", "HSV Tuner")
        self.v_upper = cv2.getTrackbarPos("V_high", "HSV Tuner")

    def _create_tracker(self):
        # Prefer CSRT (opencv-contrib), then fallbacks
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
            rospy.loginfo_once("Using legacy CSRT tracker")
            return tracker
        except Exception:
            pass
        try:
            tracker = cv2.TrackerCSRT_create()
            rospy.loginfo_once("Using CSRT tracker")
            return tracker
        except Exception:
            pass
        try:
            tracker = cv2.TrackerKCF_create()
            rospy.logwarn_once("CSRT unavailable; using KCF")
            return tracker
        except Exception:
            pass
        try:
            

            tracker = cv2.legacy.TrackerMOSSE_create()

            rospy.logwarn_once("CSRT/KCF unavailable; using MOSSE")
            return tracker
        except Exception as e:
            rospy.logerr("No OpenCV trackers available: %s", e)
            return None

    def init_tracker(self, frame, bbox):
        self.tracker = self._create_tracker()
        if self.tracker is None:
            self.tracker_ok = False
            return
        self.tracker_ok = self.tracker.init(frame, tuple(bbox))
        self.tracker_bbox = tuple(bbox)
        self.bbox_history = [tuple(bbox)]
        if self.tracker_ok:
            rospy.loginfo("Tracker initialized at bbox: %s", str(bbox))
        else:
            rospy.logwarn("Tracker init failed")

    def smooth_bbox(self, bbox):
        self.bbox_history.append(tuple(bbox))
        if len(self.bbox_history) > self.merge_history:
            self.bbox_history.pop(0)
        x_vals, y_vals, w_vals, h_vals = zip(*self.bbox_history)
        x = int(np.mean(x_vals)); y = int(np.mean(y_vals))
        w = int(np.mean(w_vals)); h = int(np.mean(h_vals))
        return (x, y, w, h)

    def _score_and_filter_contours(self, contours):
        best = None
        best_score = -1.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            aspect = w / float(h)
            if aspect < self.ar_min or aspect > self.ar_max:
                continue
            extent = area / float(w * h)

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = (area / float(hull_area)) if hull_area > 0 else 0.0

            if solidity < self.min_solidity or extent < self.min_extent:
                continue

            # Prefer slightly larger (but within cap), high solidity/extent
            area_norm = min(1.0, area / float(max(self.min_area, 1)))
            score = 0.4*area_norm + 0.35*solidity + 0.25*extent
            if score > best_score:
                best_score = score
                best = (x, y, w, h)
        return best

    def detect_green(self, frame):
        # Convert to HSV (OpenCV H:0-179, S/V:0-255)[6][12]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_lower, self.s_lower, self.v_lower], dtype=np.uint8)
        upper = np.array([self.h_upper, self.s_upper, self.v_upper], dtype=np.uint8)

        # Color mask and light cleanup
        mask = cv2.inRange(hsv, lower, upper)  # primary color segmentation[9]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        # Robust contour selection for small targets
        best = self._score_and_filter_contours(contours)
        if best is not None:
            return best, mask

        # Fallback: largest contour within area range if filtering removed all
        candidates = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in contours
                      if self.min_area <= cv2.contourArea(c) <= self.max_area]
        if candidates:
            return max(candidates, key=lambda t: t[0])[1], mask

        return None, mask

    def update_tracking(self, frame):
        now = self.frame_ts if self.frame_ts else time.time()
        force_redetect = (now - self.last_detect_time) >= self.redetect_interval or not self.tracker_ok

        detected_bbox = None
        mask = None

        if force_redetect:
            detected_bbox, mask = self.detect_green(frame)
            if detected_bbox is not None:
                self.last_detect_time = now
                self.init_tracker(frame, detected_bbox)
                return detected_bbox, mask

        if self.tracker is not None and self.tracker_ok:
            ok, bbox = self.tracker.update(frame)
            if ok:
                self.tracker_ok = True
                self.tracker_bbox = tuple(map(int, bbox))
                return self.tracker_bbox, mask
            else:
                self.tracker_ok = False  # will re-detect next loop

        if detected_bbox is None:
            detected_bbox, mask = self.detect_green(frame)
            if detected_bbox is not None:
                self.init_tracker(frame, detected_bbox)
                return detected_bbox, mask

        return None, mask

    def draw_annotations(self, frame, bbox, mask=None):
        # Mask inset (top-left) for quick debug
        if mask is not None:
            small_mask = cv2.resize(mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            small_mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
            h, w = small_mask_bgr.shape[:2]
            frame[0:h, 0:w] = small_mask_bgr

        status = "No detect"
        if bbox is not None:
            x, y, w, h = bbox
            bbox_smooth = self.smooth_bbox((x, y, w, h))
            x, y, w, h = bbox_smooth
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            status = "Detect"
            if self.tracker_ok:
                status = "Track"

        cv2.putText(frame, f"Status: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (50, 220, 50) if status != "No detect" else (0, 0, 255), 2, cv2.LINE_AA)
        return frame, (bbox if status != "No detect" else None)

    def zoom_on_bbox(self, frame, bbox, zoom_factor=1.7):
        H, W = frame.shape[:2]
        if bbox is None or zoom_factor <= 1.01:
            return frame

        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2

        crop_w = max(self.min_crop_size_px, int(W / zoom_factor))
        crop_h = max(self.min_crop_size_px, int(H / zoom_factor))

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(W, x1 + crop_w)
        y2 = min(H, y1 + crop_h)

        # Adjust if crop hits borders
        x1 = max(0, min(x1, W - (x2 - x1)))
        y1 = max(0, min(y1, H - (y2 - y1)))

        cropped = frame[y1:y2, x1:x2]

        # Upscale back to window size for display using high-quality interpolation[4][1][3]
        zoomed = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_CUBIC)
        return zoomed

    def run(self):
        cv2.namedWindow("Panda Camera Feed", cv2.WINDOW_NORMAL)
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.frame is not None:
                self._maybe_update_hsv_from_trackbars()

                frame = self.frame.copy()
                bbox, mask = self.update_tracking(frame)
                frame_out, box_for_zoom = self.draw_annotations(frame, bbox, mask)

                # Display zoom centered on the current bbox if present
                frame_out = self.zoom_on_bbox(frame_out, box_for_zoom, zoom_factor=self.zoom_factor)

                cv2.imshow("Panda Camera Feed", frame_out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            rate.sleep()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    viewer = PandaCameraViewer()
    viewer.run()
