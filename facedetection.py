# Import necessary libraries
import cv2
import time
import numpy as np
import mediapipe as mp
import threading
import os # To check for file existence

# Enum-like class for algorithm selection
class DetectionAlgorithm:
    MEDIAPIPE = "mediapipe"
    HAAR = "haar"
    LBP = "lbp"
    YUNET = "yunet"

class FaceTracker:
    """
    A reusable library class to manage camera capture, face detection using
    various algorithms (MediaPipe, Haar, LBP, YuNet), orientation adjustments,
    and displaying video feed with overlays.

    Key Functionalities:
    - Initializes and manages a camera feed.
    - Rotates the camera frame based on 'camera_native_orientation' BEFORE detection
      to create a standard-oriented frame for detection.
    - Detects faces using a selected algorithm on the standard-oriented frame.
    - Rotates the standard-oriented frame based on 'display_orientation' for final display.
    - Transforms detection results (bounding box) to match the final display orientation.
    - Calculates the offset of the detected face from the center of the display.
    - Provides options to display the video feed, center axes (lines and labels),
      and a KPI overlay (FPS, Resolution, Offset) - all text is horizontal relative to viewer.
    - Allows querying the face detection status and offset values.
    - Can run the processing loop in a separate thread.

    Main Public Methods:
    - __init__(...): Constructor to configure the tracker and select algorithm.
    - start(run_in_thread=False): Starts the tracking loop.
    - stop(): Stops the tracking loop and releases resources.
    - release(): Releases camera and closes resources manually.
    - get_status(): Returns a dictionary with detection status, offset, and FPS.
    - get_latest_frame(): Returns the latest processed frame (with overlays).
    - enable_video(show=True): Toggles the display window.
    - enable_axis(show=True): Toggles the center axis lines and labels overlay.
    - enable_kpi_overlay(show=True): Toggles the KPI overlay.
    """

    def __init__(self,
                 camera_index=0,
                 desired_width=640,
                 desired_height=480,
                 camera_native_orientation=0,
                 display_orientation=0,
                 algorithm=DetectionAlgorithm.MEDIAPIPE, # Algorithm choice
                 haar_model_path="haarcascade_frontalface_default.xml", # Path for Haar
                 lbp_model_path="lbpcascade_frontalface_improved.xml", # Path for LBP
                 yunet_model_path="face_detection_yunet_2023mar.onnx", # Path for YuNet
                 yunet_score_threshold=0.9, # Confidence threshold for YuNet
                 min_detection_confidence=0.5, # Confidence for MediaPipe
                 show_video_default=True,
                 show_axis_default=True,
                 show_kpis_default=True):
        """
        Initializes the FaceTracker.

        Args:
            camera_index (int): Index of the camera to use.
            desired_width (int): Desired width for processing/display (before rotation).
            desired_height (int): Desired height for processing/display (before rotation).
            camera_native_orientation (int): Natural orientation of the camera output (0, 90, 180, 270 deg CW).
            display_orientation (int): Desired final orientation on screen (0, 90, 180, 270 deg CW).
            algorithm (str): The detection algorithm to use (from DetectionAlgorithm enum).
            haar_model_path (str): Path to the Haar cascade XML file.
            lbp_model_path (str): Path to the LBP cascade XML file.
            yunet_model_path (str): Path to the YuNet ONNX model file.
            yunet_score_threshold (float): Confidence threshold for YuNet detections.
            min_detection_confidence (float): Minimum confidence threshold for MediaPipe face detection.
            show_video_default (bool): Whether to show the video window by default.
            show_axis_default (bool): Whether to show center axis lines and labels by default.
            show_kpis_default (bool): Whether to show KPI overlay by default.
        """
        self._init_start_time = time.time() # Track init time
        print("Initializing FaceTracker...")
        self.camera_index = camera_index
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.camera_native_orientation = camera_native_orientation % 360
        self.display_orientation = display_orientation % 360
        self.algorithm = algorithm
        self.min_detection_confidence = min_detection_confidence # MediaPipe specific
        self.yunet_score_threshold = yunet_score_threshold # YuNet specific

        # Store model paths and thresholds
        self.haar_model_path = haar_model_path
        self.lbp_model_path = lbp_model_path
        self.yunet_model_path = yunet_model_path


        # --- State Variables ---
        self.is_running = False
        self.show_video = show_video_default
        self.show_axis = show_axis_default # Controls both lines and labels
        self.show_kpis = show_kpis_default
        self.face_detected = False
        self.offset_x = None # Transformed offset X for display
        self.offset_y = None # Transformed offset Y for display
        self.latest_frame = None # Store the latest processed frame for potential external use
        self.fps = 0
        self._lock = threading.Lock() # For thread safety when accessing state
        self._thread = None
        self.detector = None # Will hold the initialized detector object

        # --- Configuration & Initialization ---
        # Init camera first to get actual dimensions if needed
        self._init_camera()
        if self.cap: # Only proceed if camera init was successful
            self._configure_orientation() # Now configure orientation based on potentially corrected dimensions
            self._init_detector() # Initialize the selected detector
            self._init_drawing_settings()

        # Check if essential components initialized successfully
        if self.cap and self.detector:
             print(f"FaceTracker initialized successfully with {self.algorithm} detector in {time.time() - self._init_start_time:.2f} seconds.")
        else:
             print("FaceTracker initialization failed.")
             self.release() # Clean up if init failed

    # --- Private Initialization Methods ---

    def _configure_orientation(self):
        """Calculates rotation needed and expected final dimensions."""
        print("Configuring orientation...")
        # Rotation to apply to camera frame to make it 'standard' orientation (0 deg) for detection
        self.cam_orient_rotation = (-self.camera_native_orientation + 360) % 360
        print(f"  Camera Native Orientation: {self.camera_native_orientation} deg")
        print(f"  Rotation applied to camera frame BEFORE detection: {self.cam_orient_rotation} deg")

        # Rotation to apply to the 'standard' oriented frame to get the final display orientation
        self.display_rotation = (-self.display_orientation) % 360 # Use the direct value
        print(f"  Desired Display Orientation: {self.display_orientation} deg")
        print(f"  Rotation applied AFTER detection for display: {self.display_rotation} deg")

        # Determine dimensions of the frame used for detection (after cam_orient_rotation)
        # Use the potentially corrected desired_width/height from _init_camera
        if self.camera_native_orientation == 90 or self.camera_native_orientation == 270:
             base_w, base_h = self.desired_height, self.desired_width
        else:
             base_w, base_h = self.desired_width, self.desired_height

        if self.cam_orient_rotation == 90 or self.cam_orient_rotation == 270:
             self.detect_frame_width = base_h
             self.detect_frame_height = base_w
        else:
             self.detect_frame_width = base_w
             self.detect_frame_height = base_h
        print(f"  Expected Detection Frame Size: {self.detect_frame_width}x{self.detect_frame_height}")


        # Determine final display dimensions based on display_rotation applied to detect_frame dimensions
        if self.display_rotation == 90 or self.display_rotation == 270:
            self.final_display_width = self.detect_frame_height
            self.final_display_height = self.detect_frame_width
        else:
            self.final_display_width = self.detect_frame_width
            self.final_display_height = self.detect_frame_height
        print(f"  Expected Final Display Size: {self.final_display_width}x{self.final_display_height}")

        # Calculate the counter-rotation needed for text overlays (relative to final display)
        self.text_rotation_angle = (self.display_rotation + 360) % 360 # Based only on display rotation

    def _init_camera(self):
        """Initializes the camera capture."""
        print("Initializing USB camera...")
        start_time = time.time()
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print(f"Error: Camera not found or failed to open at index {self.camera_index}.")
                self.cap = None
                return
            # Set resolution based on DESIRED W/H
            set_w, set_h = self.desired_width, self.desired_height
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, set_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, set_h)
            print(f"  -> Camera Initialization & Resolution Set took: {time.time() - start_time:.2f} seconds")
            self.actual_cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Actual camera capture resolution: {self.actual_cam_width}x{self.actual_cam_height}")
            # Update desired dimensions if camera didn't comply, to avoid errors later
            if self.actual_cam_width != self.desired_width or self.actual_cam_height != self.desired_height:
                 print(f"  Warning: Camera resolution mismatch. Using {self.actual_cam_width}x{self.actual_cam_height} as base.")
                 self.desired_width = self.actual_cam_width
                 self.desired_height = self.actual_cam_height
                 # Note: _configure_orientation will use these updated values

        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.cap = None

    def _init_detector(self):
        """Initializes the selected face detection model."""
        print(f"Initializing detector: {self.algorithm}...")
        start_time = time.time()
        self.detector = None
        try:
            if self.algorithm == DetectionAlgorithm.MEDIAPIPE:
                mp_face_detection = mp.solutions.face_detection
                self.detector = mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=self.min_detection_confidence)

            elif self.algorithm == DetectionAlgorithm.HAAR:
                if not os.path.exists(self.haar_model_path):
                    print(f"Error: Haar model file not found at {self.haar_model_path}")
                    return
                self.detector = cv2.CascadeClassifier(self.haar_model_path)

            elif self.algorithm == DetectionAlgorithm.LBP:
                if not os.path.exists(self.lbp_model_path):
                    print(f"Error: LBP model file not found at {self.lbp_model_path}")
                    return
                self.detector = cv2.CascadeClassifier(self.lbp_model_path)

            elif self.algorithm == DetectionAlgorithm.YUNET:
                if not os.path.exists(self.yunet_model_path):
                     print(f"Error: YuNet model file not found at {self.yunet_model_path}")
                     return
                # Initialize YuNet detector - input size will be set per-frame based on detect_frame size
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_model_path, config="", input_size=[10, 10], # Placeholder size
                    score_threshold=self.yunet_score_threshold, nms_threshold=0.3, top_k=5000
                )
            else:
                print(f"Error: Unknown detection algorithm '{self.algorithm}'")
                return

            if self.detector is not None:
                 print(f"  -> Detector Initialization took: {time.time() - start_time:.2f} seconds")
            else:
                 print(f"  -> Detector Initialization failed for {self.algorithm}.")

        except Exception as e:
            print(f"Error initializing {self.algorithm} detector: {e}")
            self.detector = None


    def _init_drawing_settings(self):
        """Initializes settings for drawing overlays."""
        print("Initializing drawing settings...")
        self.center_line_color_bgr = (255, 255, 0) # Cyan
        self.center_line_thickness = 1
        self.axis_label_color_bgr = (0, 255, 0) # Green
        self.axis_label_font_scale = 0.4
        self.axis_label_thickness = 1
        self.axis_label_interval = 100
        self.axis_label_offset = 5
        self.kpi_color_bgr = (0, 255, 0) # Green
        self.kpi_font_scale = 0.6
        self.kpi_thickness = 1
        self.kpi_margin = 10
        self.kpi_line_spacing = 5
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    # --- Private Helper Methods ---

    def _rotate_image(self, image, angle):
        """Rotates an image by 0, 90, 180, or 270 degrees clockwise."""
        angle = angle % 360
        if angle == 0: return image
        elif angle == 90:
            try: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            except AttributeError: return cv2.flip(cv2.transpose(image), 1)
        elif angle == 180:
            try: return cv2.rotate(image, cv2.ROTATE_180)
            except AttributeError: return cv2.flip(image, -1)
        elif angle == 270:
            try: return cv2.rotate(image, cv2.ROTATE_90_COUNTER_CLOCKWISE)
            except AttributeError: return cv2.flip(cv2.transpose(image), 0)
        return image

    def _transform_offset(self, offset_x, offset_y, angle):
        """Transforms raw offset coordinates to match visual orientation."""
        # This transforms offset calculated relative to the *display* frame center
        # The angle should be the display_rotation
        angle = angle % 360
        if angle == 0: return offset_x, offset_y
        elif angle == 90: return -offset_y, offset_x
        elif angle == 180: return -offset_x, -offset_y
        elif angle == 270: return offset_y, -offset_x
        return offset_x, offset_y

    def _transform_absolute_bbox(self, bbox_src, src_w, src_h, angle):
        """Transforms an absolute bounding box (xmin, ymin, xmax, ymax)
           from source coordinates to target coordinates based on rotation angle."""
        # The angle should be the display_rotation
        angle = angle % 360
        xmin_src, ymin_src, xmax_src, ymax_src = bbox_src

        if angle == 0:
            return bbox_src
        elif angle == 90: # Clockwise 90
            xmin_tgt = ymin_src
            ymin_tgt = src_w - xmax_src # Target Y depends on source W and source Xmax
            xmax_tgt = ymax_src
            ymax_tgt = src_w - xmin_src # Target Ymax depends on source W and source Xmin
            return (xmin_tgt, ymin_tgt, xmax_tgt, ymax_tgt)
        elif angle == 180: # Clockwise 180
            xmin_tgt = src_w - xmax_src
            ymin_tgt = src_h - ymax_src
            xmax_tgt = src_w - xmin_src
            ymax_tgt = src_h - ymin_src
            return (xmin_tgt, ymin_tgt, xmax_tgt, ymax_tgt)
        elif angle == 270: # Clockwise 270
            xmin_tgt = src_h - ymax_src # Target X depends on source H and source Ymax
            ymin_tgt = xmin_src
            xmax_tgt = src_h - ymin_src # Target Xmax depends on source H and source Ymin
            ymax_tgt = xmax_src
            return (xmin_tgt, ymin_tgt, xmax_tgt, ymax_tgt)
        return bbox_src # Should not happen

    def _overlay_rotated_text(self, background_img, text_img_gray, target_pos, text_color_bgr):
        """Overlays a rotated grayscale text image onto a background image."""
        target_pos = (int(target_pos[0]), int(target_pos[1]))
        rotated_text_h, rotated_text_w = text_img_gray.shape[:2]
        bg_h, bg_w = background_img.shape[:2]
        start_col = target_pos[0]
        start_row = target_pos[1]
        end_row = min(start_row + rotated_text_h, bg_h)
        end_col = min(start_col + rotated_text_w, bg_w)
        start_row = max(0, start_row)
        start_col = max(0, start_col)
        roi_h = end_row - start_row
        roi_w = end_col - start_col

        if roi_h > 0 and roi_w > 0:
            start_row_int, end_row_int = int(start_row), int(end_row)
            start_col_int, end_col_int = int(start_col), int(end_col)
            roi = background_img[start_row_int:end_row_int, start_col_int:end_col_int]
            text_mask_part = text_img_gray[:int(roi_h), :int(roi_w)]
            if roi.size == 0 or text_mask_part.size == 0: return background_img
            if len(roi.shape) == 3 and roi.shape[2] == 3:
                 mask_3channel = cv2.cvtColor(text_mask_part, cv2.COLOR_GRAY2BGR) > 0
            else: mask_3channel = text_mask_part > 0
            if mask_3channel.shape[:2] != roi.shape[:2]:
                mask_3channel = cv2.resize(mask_3channel.astype(np.uint8) * 255, (roi.shape[1], roi.shape[0])) > 0
                if len(roi.shape) == 3 and len(mask_3channel.shape) == 2:
                     mask_3channel = cv2.cvtColor(mask_3channel.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) > 0
            colored_text_block = np.zeros_like(roi)
            if len(roi.shape) == 3 and roi.shape[2] == 3: colored_text_block[:, :] = text_color_bgr
            else: colored_text_block[:, :] = int(sum(text_color_bgr) / 3)
            try:
                colored_text_masked = np.where(mask_3channel, colored_text_block, 0)
                roi_blacked_out = np.where(mask_3channel, 0, roi)
                roi_final = cv2.add(roi_blacked_out, colored_text_masked)
                background_img[start_row_int:end_row_int, start_col_int:end_col_int] = roi_final
            except ValueError as e: print(f"Error during numpy overlay: {e}"); pass
        return background_img

    def _process_frame(self):
        """Reads frame, applies camera orientation, detects face, rotates for display, transforms results."""
        if not self.cap or not self.cap.isOpened(): return None, False, None, None, None
        ret, camera_frame = self.cap.read()
        if not ret: print("Error: Failed to capture frame."); return None, False, None, None, None

        # --- Apply initial rotation based on camera orientation ---
        frame_for_detection = self._rotate_image(camera_frame, self.cam_orient_rotation)
        detect_h, detect_w = frame_for_detection.shape[:2] # Dimensions of frame used for detection

        detected = False
        offset_x = None
        offset_y = None
        absolute_bbox_detection = None # Bbox relative to frame_for_detection
        absolute_bbox_disp = None # Bbox relative to display_frame

        if self.detector:
            try:
                # --- Perform detection on the camera-oriented frame ---

                # --- MediaPipe Detection ---
                if self.algorithm == DetectionAlgorithm.MEDIAPIPE:
                    rgb_detect_frame = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
                    rgb_detect_frame.flags.writeable = False
                    results = self.detector.process(rgb_detect_frame)
                    if results.detections:
                        detection = results.detections[0]
                        bbox_relative = detection.location_data.relative_bounding_box
                        xmin = int(bbox_relative.xmin * detect_w)
                        ymin = int(bbox_relative.ymin * detect_h)
                        xmax = int((bbox_relative.xmin + bbox_relative.width) * detect_w)
                        ymax = int((bbox_relative.ymin + bbox_relative.height) * detect_h)
                        absolute_bbox_detection = (xmin, ymin, xmax, ymax)
                        detected = True

                # --- Haar/LBP Cascade Detection ---
                elif self.algorithm in [DetectionAlgorithm.HAAR, DetectionAlgorithm.LBP]:
                    gray_detect_frame = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2GRAY)
                    faces = self.detector.detectMultiScale(gray_detect_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                        x, y, w, h = faces[0]
                        absolute_bbox_detection = (x, y, x + w, y + h)
                        detected = True

                # --- YuNet Detection ---
                elif self.algorithm == DetectionAlgorithm.YUNET:
                     self.detector.setInputSize([detect_w, detect_h]) # Use detection frame size
                     results = self.detector.detect(frame_for_detection) # Use BGR detection frame
                     if results[1] is not None:
                         face_info = results[1][0]
                         x1, y1, w, h = face_info[0:4].astype(np.int32)
                         absolute_bbox_detection = (x1, y1, x1 + w, y1 + h)
                         detected = True

            except Exception as e:
                 print(f"Error during {self.algorithm} detection processing: {e}")
                 detected = False; absolute_bbox_detection = None

        # --- Rotate Frame for Final Display ---
        # Rotate the already camera-oriented frame further by display_rotation
        display_frame = self._rotate_image(frame_for_detection, self.display_rotation)
        display_height, display_width = display_frame.shape[:2]
        frame_center_x = display_width // 2
        frame_center_y = display_height // 2

        # --- Transform Bbox and Calculate Offset (if detected) ---
        if detected and absolute_bbox_detection:
            # Transform the bbox from detection frame coords to display frame coords
            absolute_bbox_disp = self._transform_absolute_bbox(
                absolute_bbox_detection, detect_w, detect_h, self.display_orientation # Use display_rotation here
            )

            # Clamp transformed coordinates
            xmin, ymin, xmax, ymax = absolute_bbox_disp
            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(display_width - 1, xmax); ymax = min(display_height - 1, ymax)
            absolute_bbox_disp = (xmin, ymin, xmax, ymax) # Update clamped bbox

            # Calculate center and offset relative to display frame center
            face_center_x = (xmin + xmax) // 2
            face_center_y = (ymin + ymax) // 2
            original_offset_x = face_center_x - frame_center_x
            original_offset_y = face_center_y - frame_center_y
            # Transform offset values to match the visual meaning on the display
            offset_x, offset_y = self._transform_offset(
                original_offset_x, original_offset_y, self.display_orientation # Use display_rotation here
            )

        return display_frame, detected, offset_x, offset_y, absolute_bbox_disp # Return bbox for display

    def _draw_overlays(self, frame, detected, offset_x, offset_y, absolute_bbox_disp):
        """Draws all enabled overlays onto the frame."""
        if frame is None: return None
        display_height, display_width = frame.shape[:2]
        frame_center_x = display_width // 2
        frame_center_y = display_height // 2

        # --- Draw Center Lines and Axis Labels (if enabled) ---
        if self.show_axis:
            # Draw Center Lines
            cv2.line(frame, (frame_center_x, 0), (frame_center_x, display_height - 1), self.center_line_color_bgr, self.center_line_thickness)
            cv2.line(frame, (0, frame_center_y), (display_width - 1, frame_center_y), self.center_line_color_bgr, self.center_line_thickness)

            # Draw Axis Labels (counter-rotated text)
            # Determine which *original* axis corresponds to the display axis based on display_rotation
            if self.display_rotation == 0 or self.display_rotation == 180:
                horiz_prefix = "X"; vert_prefix = "Y"
            else: # 90 or 270
                horiz_prefix = "Y"; vert_prefix = "X"

            # Draw labels along the display's horizontal center line
            for x_abs in range(0, display_width, self.axis_label_interval):
                 # Calculate the value this position represents on the *original* axis
                if horiz_prefix == "X":
                    coord_rel = x_abs - frame_center_x
                    if self.display_rotation == 180: coord_rel *= -1 # Invert X for 180
                else: # Horizontal axis represents Y
                    coord_rel = frame_center_y - x_abs # Inverted Y relative to display X
                    if self.display_rotation == 90: coord_rel *= -1 # Invert Y value for 90 deg CW display

                if x_abs != frame_center_x:
                    txt = f"{horiz_prefix}:{coord_rel}"
                    (tw, th), bl = cv2.getTextSize(txt, self.font, self.axis_label_font_scale, self.axis_label_thickness)
                    img = np.zeros((th + bl, tw), dtype=np.uint8); cv2.putText(img, txt, (0, th), self.font, self.axis_label_font_scale, (255), self.axis_label_thickness, self.line_type)
                    r_img = self._rotate_image(img, self.text_rotation_angle); rh, rw = r_img.shape[:2]
                    tx = x_abs - rw // 2; ty = frame_center_y - self.axis_label_offset - rh
                    frame = self._overlay_rotated_text(frame, r_img, (tx, ty), self.axis_label_color_bgr)

            # Draw labels along the display's vertical center line
            for y_abs in range(0, display_height, self.axis_label_interval):
                # Calculate the value this position represents on the *original* axis
                if vert_prefix == "Y":
                    coord_rel = frame_center_y - y_abs # Inverted Y relative to display Y
                    if self.display_rotation == 180: coord_rel *= -1 # Invert Y for 180
                else: # Vertical axis represents X
                    coord_rel = y_abs - frame_center_x
                    if self.display_rotation == 270: coord_rel *= -1 # Invert X for 270 deg CW display

                if y_abs != frame_center_y:
                    txt = f"{vert_prefix}:{coord_rel}"
                    (tw, th), bl = cv2.getTextSize(txt, self.font, self.axis_label_font_scale, self.axis_label_thickness)
                    img = np.zeros((th + bl, tw), dtype=np.uint8); cv2.putText(img, txt, (0, th), self.font, self.axis_label_font_scale, (255), self.axis_label_thickness, self.line_type)
                    r_img = self._rotate_image(img, self.text_rotation_angle); rh, rw = r_img.shape[:2]
                    tx = frame_center_x - self.axis_label_offset - rw; ty = y_abs - rh // 2
                    frame = self._overlay_rotated_text(frame, r_img, (tx, ty), self.axis_label_color_bgr)

        # --- Draw KPI Overlay (if enabled) ---
        if self.show_kpis:
            fps_text = f"FPS: {self.fps:.1f}"
            res_text = f"Res: {display_width}x{display_height}"
            off_x_txt = f"Off X: {offset_x:.1f}" if offset_x is not None else "Off X: N/A"
            off_y_txt = f"Off Y: {offset_y:.1f}" if offset_y is not None else "Off Y: N/A"
            texts = [fps_text, res_text, off_x_txt, off_y_txt]
            dims = [cv2.getTextSize(t, self.font, self.kpi_font_scale, self.kpi_thickness) for t in texts]
            ths = [d[0][1] for d in dims]; bls = [d[1] for d in dims]
            max_w = max(d[0][0] for d in dims) if dims else 0
            if max_w > 0:
                blk_h = sum(ths) + (self.kpi_line_spacing * (len(texts))) + bls[-1]
                kpi_img = np.zeros((blk_h, max_w), dtype=np.uint8)
                curr_y = ths[0]
                for i, txt in enumerate(texts):
                    cv2.putText(kpi_img, txt, (0, curr_y), self.font, self.kpi_font_scale, (255), self.kpi_thickness, self.line_type)
                    if i < len(texts) - 1: curr_y += ths[i] + self.kpi_line_spacing
                r_kpi_img = self._rotate_image(kpi_img, self.text_rotation_angle)
                frame = self._overlay_rotated_text(frame, r_kpi_img, (self.kpi_margin, self.kpi_margin), self.kpi_color_bgr)

        # --- Draw Face BBox (if detected) ---
        # Use the absolute_bbox_disp which is already transformed for the display frame
        if detected and absolute_bbox_disp:
            bbox_color = (0, 255, 0) # Green
            xmin, ymin, xmax, ymax = [int(c) for c in absolute_bbox_disp] # Ensure integer coords
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), bbox_color, 2)
            # Skip center circle drawing for simplicity

        return frame

    # --- Public Control Methods ---

    def start(self, run_in_thread=False):
        """Starts the face tracking loop."""
        if self.is_running: print("Tracker is already running."); return
        if not self.cap or not self.detector: print("Error: Cannot start tracker, initialization failed."); return
        self.is_running = True
        if run_in_thread:
            self._thread = threading.Thread(target=self._run_loop, daemon=True); self._thread.start()
            print("Tracker started in background thread.")
        else: print("Tracker started in main thread (blocking)."); self._run_loop()

    def stop(self):
        """Stops the face tracking loop and releases resources."""
        if not self.is_running: print("Tracker is not running."); return
        print("Stopping tracker..."); self.is_running = False
        if self._thread and self._thread.is_alive():
            print("Waiting for background thread to finish..."); self._thread.join(timeout=2)
            if self._thread.is_alive(): print("Warning: Background thread did not stop gracefully.")
        self._thread = None; print("Releasing resources..."); self.release(); print("Tracker stopped.")

    def release(self):
        """Releases camera and closes detector instance."""
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened(): print("Releasing camera..."); self.cap.release(); self.cap = None
        # MediaPipe needs explicit close, others don't have a specific close method
        if self.algorithm == DetectionAlgorithm.MEDIAPIPE and hasattr(self, 'detector') and self.detector:
             print("Closing MediaPipe detector..."); self.detector.close()
        self.detector = None # Clear reference for all types
        print("Closing OpenCV windows..."); cv2.destroyAllWindows()

    def get_status(self):
        """Returns the latest face detection status and offset."""
        with self._lock: return {"detected": self.face_detected, "offset_x": self.offset_x, "offset_y": self.offset_y, "fps": self.fps}

    def get_latest_frame(self):
        """Returns the latest processed frame with overlays."""
        with self._lock: return self.latest_frame.copy() if self.latest_frame is not None else None

    def enable_video(self, show=True):
        """Enable or disable the video display window."""
        self.show_video = show

    def enable_axis(self, show=True):
        """Enable or disable the center axis lines and labels overlay."""
        # This method now controls the flag for both lines and labels
        self.show_axis = show

    def enable_kpi_overlay(self, show=True):
        """Enable or disable the KPI text overlay."""
        self.show_kpis = show

    # --- Private Main Loop ---

    def _run_loop(self):
        """The main processing and display loop."""
        prev_frame_time = 0
        window_name = "Face Tracker Feed"
        if self.show_video:
            print("Setting up display window...")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.final_display_width, self.final_display_height)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.is_running:
            loop_start_time = time.time()
            # Process frame now returns bbox relative to display frame
            display_frame, detected, offset_x, offset_y, absolute_bbox_disp = self._process_frame()
            if display_frame is None:
                if not self.cap or not self.cap.isOpened(): print("Camera disconnected or failed. Stopping."); self.is_running = False; break
                time.sleep(0.1); continue
            with self._lock:
                self.face_detected = detected; self.offset_x = offset_x; self.offset_y = offset_y
                if prev_frame_time > 0: loop_time = loop_start_time - prev_frame_time; self.fps = 1.0 / loop_time if loop_time > 0 else 0
                prev_frame_time = loop_start_time
            frame_to_show = display_frame.copy()
            # Pass the display-relative bbox to draw_overlays
            frame_to_show = self._draw_overlays(frame_to_show, detected, offset_x, offset_y, absolute_bbox_disp)
            with self._lock: self.latest_frame = frame_to_show
            if self.show_video:
                try:
                    cv2.imshow(window_name, frame_to_show)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): print("'q' pressed, stopping."); self.is_running = False
                except cv2.error as e:
                     if "NULL window" in str(e) or "Invalid window" in str(e): print("Display window closed externally. Stopping video display."); self.show_video = False
                     else: print(f"OpenCV display error: {e}"); self.is_running = False
            else: time.sleep(0.01)
        print("Exiting run loop.")


# --- Example Usage ---
if __name__ == "__main__":
    print("Running FaceTracker Example...")

    # --- Configuration ---
    CAM_IDX = 0
    WIDTH = 640
    HEIGHT = 480
    CAM_ORIENT = 180 # Camera is upside down
    DISP_ORIENT = 90  # Display portrait CW

    # --- CHOOSE ALGORITHM ---
    ALGO = DetectionAlgorithm.MEDIAPIPE
    # ALGO = DetectionAlgorithm.HAAR
    # ALGO = DetectionAlgorithm.LBP
    # ALGO = DetectionAlgorithm.YUNET # Make sure ONNX file exists

    # --- Instantiate ---
    print(f"\n--- Initializing with {ALGO} ---")
    tracker = FaceTracker(
        camera_index=CAM_IDX,
        desired_width=WIDTH,
        desired_height=HEIGHT,
        camera_native_orientation=CAM_ORIENT,
        display_orientation=DISP_ORIENT,
        algorithm=ALGO, # Pass selected algorithm
        # Provide correct paths if models are not in the same directory
        # haar_model_path="path/to/haarcascade_frontalface_default.xml",
        # lbp_model_path="path/to/lbpcascade_frontalface_improved.xml",
        yunet_model_path="face_detection_yunet_2023mar.onnx", # Ensure this exists
        show_video_default=True,
        show_axis_default=True,
        show_kpis_default=True
    )

    example_start_time = time.time()

    if tracker.cap and tracker.detector: # Check if init was successful
        tracker.start(run_in_thread=True)
        toggled_10s, toggled_15s, toggled_20s = False, False, False # Flags for toggling example
        try:
            while tracker.is_running:
                status = tracker.get_status()
                # print(f"Status: Detected={status['detected']}, X={status['offset_x']}, Y={status['offset_y']}, FPS={status['fps']:.1f}")
                elapsed_time = int(time.time() - example_start_time)
                if elapsed_time >= 10 and not toggled_10s: print("\nDisabling Axis Lines & Labels after 10s\n"); tracker.enable_axis(False); toggled_10s = True # Use enable_axis
                if elapsed_time >= 15 and not toggled_15s: print("\nDisabling KPI Overlay after 15s\n"); tracker.enable_kpi_overlay(False); toggled_15s = True
                if elapsed_time >= 20 and not toggled_20s: print("\nRe-enabling overlays after 20s\n"); tracker.enable_axis(True); tracker.enable_kpi_overlay(True); toggled_20s = True # Use enable_axis
                time.sleep(1)
        except KeyboardInterrupt: print("\nCtrl+C detected in main thread.")
        finally: tracker.stop()
    else:
        print("Could not initialize tracker. Exiting.")

    print("FaceTracker Example Finished.")

