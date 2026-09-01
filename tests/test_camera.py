"""
Camera Diagnostic & Hardware Initialization Test Suite
Tests camera initialization, headless environments, frame capture mock fallbacks,
and resource release guarantees.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import camera_test


class TestCameraDiagnostics(unittest.TestCase):
    """Tier 3: Mock-driven hardware diagnostic checks for camera initialization."""

    @patch('cv2.destroyAllWindows')
    @patch('cv2.waitKey')
    @patch('cv2.imshow')
    @patch('cv2.VideoCapture')
    def test_camera_successful_capture_and_release(self, mock_videocapture, mock_imshow, mock_waitkey, mock_destroy):
        """Test successful camera open, frame capture, and cleanup on exit."""
        # Setup mock capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_videocapture.return_value = mock_cap

        # Simulate user pressing 'q' on first frame
        mock_waitkey.return_value = ord('q')

        # Run diagnostic
        camera_test.test_camera()

        # Verifications
        mock_videocapture.assert_called_once_with(0)
        mock_cap.isOpened.assert_called_once()
        mock_cap.read.assert_called()
        mock_imshow.assert_called_with('Camera Test', dummy_frame)
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_camera_failed_to_open(self, mock_videocapture):
        """Test graceful error handling when camera device is inaccessible or absent."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture.return_value = mock_cap

        # Should return gracefully without throwing unhandled exceptions
        camera_test.test_camera()

        mock_cap.isOpened.assert_called_once()
        mock_cap.read.assert_not_called()

    @patch('cv2.destroyAllWindows')
    @patch('cv2.VideoCapture')
    def test_camera_frame_capture_failure(self, mock_videocapture, mock_destroy):
        """Test graceful exit when frame read fails during streaming."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_videocapture.return_value = mock_cap

        camera_test.test_camera()

        mock_cap.read.assert_called_once()
        mock_cap.release.assert_called_once()
        mock_destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
