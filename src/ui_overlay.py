"""
UI Overlay Module for Sign Language Detection ML System

Provides a high-performance, decoupled OpenCV HUD (Heads-Up Display) renderer
driven by a structured HUDState dataclass. Implements sub-array ROI alpha-blending
to achieve <0.05ms overlay latency without full-frame memory copies.
"""

from dataclasses import dataclass, field
import math
import numpy as np
import cv2


# ── Color Palette (BGR format for OpenCV) ────────────────────────────────
COL_BG        = (15, 15, 30)         # Dark navy background for transparent panels
COL_CYAN      = (200, 220, 0)        # Vibrant cyan accent
COL_AMBER     = (0, 165, 255)        # Amber / analyzing / warnings
COL_GREEN     = (0, 220, 100)        # Emerald green for detected / high confidence
COL_RED       = (60, 60, 220)        # Muted red for low confidence
COL_WHITE     = (240, 240, 240)      # High-contrast white text
COL_GREY      = (140, 140, 140)      # Controls / secondary text
COL_DIM       = (80, 80, 90)         # Borders and divider lines
COL_BAR_BG    = (40, 40, 55)         # Confidence bar background track
COL_BAR_FILL  = (200, 220, 0)        # Cyan bar fill
COL_JOINT     = (200, 220, 0)        # Landmark joints
COL_CONN      = (160, 120, 0)        # Skeleton connection lines
COL_PULSE_A   = (200, 220, 0)        # Cyan pulse phase A
COL_PULSE_B   = (0, 220, 200)        # Teal pulse phase B

# Standard MediaPipe 21-landmark hand skeletal connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


@dataclass
class HUDState:
    """Decoupled HUD state container encapsulating all rendering parameters."""
    fps: float = 0.0
    detected: bool = False
    status_text: str = "SCANNING"  # "SCANNING" | "DETECTED" | "UNCERTAIN"
    gesture: str = "None"
    confidence: float = 0.0
    sequence: list[str] = field(default_factory=list)
    hands_info: list[dict] = field(default_factory=list)  # list of {bbox, landmarks, handedness, score}
    classes: list[str] = field(default_factory=list)
    sequence_progress: float = 0.0  # 0.0 to 1.0 (countdown progress before sequence timeout)


class SignLanguageHUD:
    """
    Decoupled OpenCV HUD Renderer implementing high-performance sub-array
    ROI alpha blending and premium visual elements.
    """

    def __init__(self, class_names: list[str] | None = None):
        """
        Initialize the HUD renderer.

        Args:
            class_names (list[str] | None): Optional list of gesture class names for legend.
        """
        self.class_names = class_names or []
        self._frame_count = 0

    @staticmethod
    def blend_roi(image: np.ndarray, x: int, y: int, w: int, h: int,
                  colour: tuple[int, int, int] = COL_BG, alpha: float = 0.80) -> None:
        """
        High-performance sub-array ROI alpha blending in-place.
        Clamps bounds to frame dimensions and applies cv2.addWeighted directly on the slice,
        eliminating full-frame duplication (executes in <0.05ms).
        """
        img_h, img_w = image.shape[:2]
        x1 = max(0, min(int(x), img_w))
        y1 = max(0, min(int(y), img_h))
        x2 = max(0, min(int(x + w), img_w))
        y2 = max(0, min(int(y + h), img_h))

        if x2 <= x1 or y2 <= y1:
            return

        roi = image[y1:y2, x1:x2]
        overlay_roi = np.full_like(roi, colour, dtype=np.uint8)
        cv2.addWeighted(overlay_roi, alpha, roi, 1.0 - alpha, 0, dst=roi)

    def _overlay_rect(self, image: np.ndarray, x: int, y: int, w: int, h: int,
                      colour: tuple[int, int, int] = COL_BG, alpha: float = 0.80) -> None:
        """Backward-compatible wrapper for fast sub-array ROI blending."""
        self.blend_roi(image, x, y, w, h, colour, alpha)

    def _draw_rounded_rect(self, image: np.ndarray, x: int, y: int, w: int, h: int,
                           colour: tuple[int, int, int], thickness: int = 1, radius: int = 8) -> None:
        """Draw a rounded rectangle border."""
        x, y, w, h = int(x), int(y), int(w), int(h)
        radius = min(radius, w // 2, h // 2)
        if radius <= 0:
            cv2.rectangle(image, (x, y), (x + w, y + h), colour, thickness)
            return

        # Top-left corner
        cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, colour, thickness)
        # Top-right corner
        cv2.ellipse(image, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, colour, thickness)
        # Bottom-right corner
        cv2.ellipse(image, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, colour, thickness)
        # Bottom-left corner
        cv2.ellipse(image, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, colour, thickness)
        # Boundary lines
        cv2.line(image, (x + radius, y), (x + w - radius, y), colour, thickness)
        cv2.line(image, (x + radius, y + h), (x + w - radius, y + h), colour, thickness)
        cv2.line(image, (x, y + radius), (x, y + h - radius), colour, thickness)
        cv2.line(image, (x + w, y + radius), (x + w, y + h - radius), colour, thickness)

    def _draw_corner_brackets(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                              colour: tuple[int, int, int] = COL_CYAN, length: int = 18, thickness: int = 2) -> None:
        """Draw sleek corner brackets around a bounding box."""
        w = x2 - x1
        h = y2 - y1
        cl_x = min(length, max(4, w // 4))
        cl_y = min(length, max(4, h // 4))

        # Top-Left
        cv2.line(image, (x1, y1), (x1 + cl_x, y1), colour, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1), (x1, y1 + cl_y), colour, thickness, cv2.LINE_AA)

        # Top-Right
        cv2.line(image, (x2, y1), (x2 - cl_x, y1), colour, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1), (x2, y1 + cl_y), colour, thickness, cv2.LINE_AA)

        # Bottom-Left
        cv2.line(image, (x1, y2), (x1 + cl_x, y2), colour, thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y2), (x1, y2 - cl_y), colour, thickness, cv2.LINE_AA)

        # Bottom-Right
        cv2.line(image, (x2, y2), (x2 - cl_x, y2), colour, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y2), (x2, y2 - cl_y), colour, thickness, cv2.LINE_AA)

    def _draw_handedness_badge(self, image: np.ndarray, x: int, y: int,
                               label: str = "Right", score: float | None = None) -> None:
        """Draw a sleek translucent handedness pill badge."""
        label_str = str(label) if label is not None else "Hand"
        is_valid_score = (score is not None and isinstance(score, (int, float)) and math.isfinite(score))
        if is_valid_score:
            clamped_score = max(0.0, min(float(score), 1.0))
            text = f"{label_str} ({int(clamped_score * 100)}%)"
        else:
            text = label_str

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

        bw = tw + 20
        bh = th + 10
        bx = max(10, x)
        by = max(55, y - bh - 6)

        self.blend_roi(image, bx, by, bw, bh, COL_BG, 0.82)
        cv2.rectangle(image, (bx, by), (bx + bw, by + bh), COL_DIM, 1)

        # Indicator dot
        dot_col = COL_CYAN if label_str == "Left" else COL_AMBER
        cv2.circle(image, (bx + 7, by + bh // 2), 3, dot_col, -1, cv2.LINE_AA)
        # Text
        cv2.putText(image, text, (bx + 14, by + th + 4), font, scale, COL_WHITE, thick, cv2.LINE_AA)

    def draw_confidence_bar(self, image: np.ndarray, x: int, y: int, w: int, h: int, confidence: float | None) -> None:
        """Draw a styled confidence progress bar with threshold tick and color grading."""
        x, y, w, h = int(x), int(y), int(w), int(h)
        # Background track
        cv2.rectangle(image, (x, y), (x + w, y + h), COL_BAR_BG, -1)

        # Fill calculation
        if confidence is None:
            clamped_conf = 0.0
        else:
            try:
                conf_f = float(confidence)
                if math.isnan(conf_f) or conf_f < 0.0:
                    clamped_conf = 0.0
                elif math.isinf(conf_f) or conf_f > 1.0:
                    clamped_conf = 1.0
                else:
                    clamped_conf = conf_f
            except (ValueError, TypeError):
                clamped_conf = 0.0

        fill_w = int(w * clamped_conf)

        if clamped_conf >= 0.70:
            fill_col = COL_GREEN
        elif clamped_conf >= 0.40:
            fill_col = COL_AMBER
        else:
            fill_col = COL_RED

        if fill_w > 0:
            cv2.rectangle(image, (x, y), (x + fill_w, y + h), fill_col, -1)

        # Threshold marker at 70% activation line
        thresh_x = x + int(w * 0.70)
        cv2.line(image, (thresh_x, y - 1), (thresh_x, y + h + 1), COL_WHITE, 1)

        # Border
        cv2.rectangle(image, (x, y), (x + w, y + h), COL_DIM, 1)

        # Percentage text
        pct_text = f"{int(clamped_conf * 100)}%"
        cv2.putText(image, pct_text, (x + w + 8, y + h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_WHITE, 1, cv2.LINE_AA)

    def draw_hand_skeleton(self, image: np.ndarray, hand_landmarks) -> None:
        """Draw custom-colored hand skeleton with styled joints and connections."""
        if hand_landmarks is None:
            return

        h, w, _ = image.shape

        # Extract 2D pixel coordinates for all 21 landmarks
        pts = []
        lm_list = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks
        if not lm_list:
            return

        for lm in lm_list:
            if isinstance(lm, dict):
                pts.append((int(lm.get('x', 0) * w), int(lm.get('y', 0) * h)))
            elif hasattr(lm, 'x') and hasattr(lm, 'y'):
                pts.append((int(lm.x * w), int(lm.y * h)))

        if len(pts) < 21:
            return

        # Draw connections
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(image, pts[start_idx], pts[end_idx], COL_CONN, 2, cv2.LINE_AA)

        # Draw joints on top
        for i, (cx, cy) in enumerate(pts):
            if i in [4, 8, 12, 16, 20]:
                # Fingertips get prominent dual circles
                cv2.circle(image, (cx, cy), 6, COL_JOINT, -1, cv2.LINE_AA)
                cv2.circle(image, (cx, cy), 6, COL_WHITE, 1, cv2.LINE_AA)
            elif i == 0:
                # Wrist gets amber accent
                cv2.circle(image, (cx, cy), 5, COL_AMBER, -1, cv2.LINE_AA)
            else:
                cv2.circle(image, (cx, cy), 4, COL_JOINT, -1, cv2.LINE_AA)

    def draw_hands(self, image: np.ndarray, hands_info: list[dict]) -> None:
        """Draw skeletons, corner-bracket bounding boxes, and handedness badges for all hands."""
        h_img, w_img, _ = image.shape

        for hand in hands_info:
            lm = hand.get('landmarks')
            if lm is not None:
                self.draw_hand_skeleton(image, lm)

            # Bounding box
            bbox = hand.get('bbox')
            if bbox is None and lm is not None:
                # Compute bounding box from landmarks
                lm_list = lm.landmark if hasattr(lm, 'landmark') else lm
                if lm_list:
                    xs = [p['x'] if isinstance(p, dict) else p.x for p in lm_list if (isinstance(p, dict) and 'x' in p) or hasattr(p, 'x')]
                    ys = [p['y'] if isinstance(p, dict) else p.y for p in lm_list if (isinstance(p, dict) and 'y' in p) or hasattr(p, 'y')]
                    if xs and ys:
                        pad = 14
                        x1 = max(0, int(min(xs) * w_img) - pad)
                        y1 = max(0, int(min(ys) * h_img) - pad)
                        x2 = min(w_img, int(max(xs) * w_img) + pad)
                        y2 = min(h_img, int(max(ys) * h_img) + pad)
                        bbox = (x1, y1, x2, y2)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                self._draw_corner_brackets(image, x1, y1, x2, y2, COL_CYAN, length=18, thickness=2)

                # Handedness badge
                label = hand.get('handedness')
                score = hand.get('score')
                if label:
                    self._draw_handedness_badge(image, x1, y1, label=label, score=score)

    def draw_top_bar(self, image: np.ndarray, state_or_fps: float | HUDState | None) -> None:
        """Draw top HUD header bar with title and FPS readout."""
        h, w, _ = image.shape
        bar_h = 45
        self.blend_roi(image, 0, 0, w, bar_h, COL_BG, 0.80)

        # Title
        cv2.putText(image, "Sign Language AI", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COL_CYAN, 2, cv2.LINE_AA)

        # Subtitle / system badge
        cv2.putText(image, "[ Real-time Gesture Engine ]", (260, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GREY, 1, cv2.LINE_AA)

        # FPS calculation
        raw_fps = state_or_fps.fps if isinstance(state_or_fps, HUDState) else state_or_fps
        if raw_fps is None:
            fps_val = 0.0
        else:
            try:
                fps_val = float(raw_fps)
            except (ValueError, TypeError):
                fps_val = 0.0

        if math.isfinite(fps_val):
            fps_text = f"FPS: {fps_val:.0f}"
        else:
            fps_text = f"FPS: {fps_val}"

        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(image, fps_text, (w - fps_size[0] - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GREEN, 1, cv2.LINE_AA)

        # Divider line
        cv2.line(image, (0, bar_h), (w, bar_h), COL_DIM, 1)

    def draw_detection_panel(self, image: np.ndarray, *args, **kwargs) -> None:
        """
        Draw detection result panel.
        Supports both signatures:
          - draw_detection_panel(image, state: HUDState)
          - draw_detection_panel(image, prediction_text: str, confidence: float, status: str)
        """
        if args and isinstance(args[0], HUDState):
            state = args[0]
            status = str(state.status_text) if state.status_text is not None else "SCANNING"
            confidence = 0.0 if state.confidence is None else state.confidence
            if status == "DETECTED":
                prediction_text = str(state.gesture).upper() if state.gesture is not None else "NONE"
            elif status == "UNCERTAIN":
                prediction_text = "Analysing..."
            else:
                prediction_text = "Scanning..."
        else:
            prediction_text = args[0] if len(args) > 0 else kwargs.get('prediction_text', "None")
            confidence = args[1] if len(args) > 1 else kwargs.get('confidence', 0.0)
            status = args[2] if len(args) > 2 else kwargs.get('status', "SCANNING")
            if confidence is None:
                confidence = 0.0
            status = str(status) if status is not None else "SCANNING"

        h, w, _ = image.shape
        panel_w = w - 200
        panel_h = 60
        panel_x = 15
        panel_y = h - 150

        self.blend_roi(image, panel_x, panel_y, panel_w, panel_h, COL_BG, 0.82)

        # Border styling with pulse on detection
        if status == "DETECTED":
            pulse = COL_PULSE_A if (self._frame_count // 8) % 2 == 0 else COL_PULSE_B
            self._draw_rounded_rect(image, panel_x, panel_y, panel_w, panel_h, pulse, thickness=2)
        else:
            self._draw_rounded_rect(image, panel_x, panel_y, panel_w, panel_h, COL_DIM, thickness=1)

        # Status Indicator Dot
        if status == "DETECTED":
            dot_col = COL_GREEN
        elif status == "UNCERTAIN":
            dot_col = COL_AMBER
        else:
            dot_col = COL_DIM

        cv2.circle(image, (panel_x + 18, panel_y + 25), 6, dot_col, -1, cv2.LINE_AA)
        cv2.circle(image, (panel_x + 18, panel_y + 25), 6, COL_WHITE, 1, cv2.LINE_AA)

        # Status label
        cv2.putText(image, status, (panel_x + 32, panel_y + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, dot_col, 1, cv2.LINE_AA)

        # Prediction text
        cv2.putText(image, str(prediction_text), (panel_x + 18, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_WHITE, 1, cv2.LINE_AA)

        # Confidence Bar
        bar_x = panel_x + panel_w - 260
        bar_y = panel_y + 15
        self.draw_confidence_bar(image, bar_x, bar_y, 180, 16, confidence)

    def draw_sequence_panel(self, image: np.ndarray, *args, **kwargs) -> None:
        """
        Draw sequence text box and countdown progress line.
        Supports both signatures:
          - draw_sequence_panel(image, state: HUDState)
          - draw_sequence_panel(image, sequence_text: str)
        """
        seq_progress = 0.0
        if args and isinstance(args[0], HUDState):
            state = args[0]
            seq_items = [str(item) for item in state.sequence] if state.sequence else []
            sequence_text = '  >  '.join(seq_items) if seq_items else ""
            if state.sequence_progress is not None:
                try:
                    sp = float(state.sequence_progress)
                    seq_progress = sp if math.isfinite(sp) else 0.0
                except (ValueError, TypeError):
                    seq_progress = 0.0
        else:
            sequence_text = args[0] if len(args) > 0 else kwargs.get('sequence_text', "")

        h, w, _ = image.shape
        panel_w = w - 200
        panel_h = 40
        panel_x = 15
        panel_y = h - 85

        self.blend_roi(image, panel_x, panel_y, panel_w, panel_h, COL_BG, 0.78)
        self._draw_rounded_rect(image, panel_x, panel_y, panel_w, panel_h, COL_DIM, 1)

        # Label
        cv2.putText(image, "SEQUENCE:", (panel_x + 12, panel_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_AMBER, 1, cv2.LINE_AA)

        if sequence_text:
            cv2.putText(image, str(sequence_text), (panel_x + 110, panel_y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_WHITE, 1, cv2.LINE_AA)
        else:
            cv2.putText(image, "(waiting for gestures...)",
                        (panel_x + 110, panel_y + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_DIM, 1, cv2.LINE_AA)

        # Timeout countdown progress bar
        if seq_progress > 0.0:
            prog_w = int((panel_w - 24) * max(0.0, min(seq_progress, 1.0)))
            if prog_w > 0:
                cv2.line(image, (panel_x + 12, panel_y + panel_h - 4),
                         (panel_x + 12 + prog_w, panel_y + panel_h - 4),
                         COL_AMBER, 2, cv2.LINE_AA)

    def draw_gesture_legend(self, image: np.ndarray, *args, **kwargs) -> None:
        """
        Draw sidebar listing loaded gesture classes with active item highlight.
        Supports both signatures:
          - draw_gesture_legend(image, state: HUDState)
          - draw_gesture_legend(image, x: int, y: int)
        """
        active_gesture = None
        classes = self.class_names

        if args and isinstance(args[0], HUDState):
            state = args[0]
            classes = state.classes if (state.classes is not None and len(state.classes) > 0) else self.class_names
            if state.status_text == "DETECTED" and state.gesture is not None:
                active_gesture = str(state.gesture).lower()
            h, w, _ = image.shape
            x = w - 175
            y = 55
        else:
            x = args[0] if len(args) > 0 else kwargs.get('x', 1280 - 175)
            y = args[1] if len(args) > 1 else kwargs.get('y', 55)

        if not classes:
            return

        pad = 10
        line_h = 24
        title_h = 30
        panel_h = title_h + len(classes) * line_h + pad
        panel_w = 160

        self.blend_roi(image, x, y, panel_w, panel_h, COL_BG, 0.75)
        self._draw_rounded_rect(image, x, y, panel_w, panel_h, COL_DIM, 1)

        # Title
        cv2.putText(image, "GESTURES", (x + pad, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 1, cv2.LINE_AA)

        # Class list
        for i, name in enumerate(classes):
            ty = y + title_h + i * line_h + 16
            name_str = str(name)
            is_active = (active_gesture is not None and name_str.lower() == active_gesture)

            if is_active:
                # Glowing active bullet and highlight
                cv2.circle(image, (x + pad + 4, ty - 4), 4, COL_CYAN, -1, cv2.LINE_AA)
                cv2.putText(image, name_str.capitalize(), (x + pad + 14, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_CYAN, 2, cv2.LINE_AA)
            else:
                cv2.circle(image, (x + pad + 4, ty - 4), 3, COL_GREEN, -1, cv2.LINE_AA)
                cv2.putText(image, name_str.capitalize(), (x + pad + 14, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_WHITE, 1, cv2.LINE_AA)

    def draw_controls_bar(self, image: np.ndarray) -> None:
        """Draw bottom keyboard controls footer bar."""
        h, w, _ = image.shape
        bar_h = 30
        bar_y = h - bar_h
        self.blend_roi(image, 0, bar_y, w, bar_h, COL_BG, 0.85)
        cv2.line(image, (0, bar_y), (w, bar_y), COL_DIM, 1)

        controls = "[Q] Quit    [C] Clear Sequence    [S] Screenshot"
        cv2.putText(image, controls, (15, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_GREY, 1, cv2.LINE_AA)

    def render(self, frame: np.ndarray, state: HUDState) -> np.ndarray:
        """
        Composite all HUD overlay elements onto the video frame in-place.

        Args:
            frame (np.ndarray): Input video frame (BGR).
            state (HUDState): Encapsulated HUD state container.

        Returns:
            np.ndarray: Modified frame with complete HUD overlay.
        """
        self._frame_count += 1

        # 1. Hand skeleton annotations, corner brackets, and handedness badges
        if state.hands_info:
            self.draw_hands(frame, state.hands_info)

        # 2. Top header bar with FPS and branding
        self.draw_top_bar(frame, state)

        # 3. Detection result panel
        self.draw_detection_panel(frame, state)

        # 4. Sequence buffer panel with timeout countdown
        self.draw_sequence_panel(frame, state)

        # 5. Controls footer bar
        self.draw_controls_bar(frame)

        # 6. Gesture legend sidebar
        self.draw_gesture_legend(frame, state)

        return frame
