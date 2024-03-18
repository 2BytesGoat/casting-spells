import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui

import utils
from screen_utils import ScreenUtils

class CastingSpells:
    def __init__(self, resolution=(800, 600), min_circle_area=350):
        self.hsv_threshold_lower = np.array([0, 0, 30])
        self.hsv_threshold_upper = np.array([179, 150, 168])

        # WARNING: min_area depends on the video resolution
        self.resolution = resolution
        self.min_circle_area = min_circle_area

        self.video_path = None
        self.video = None
        self.move_positions = []

    def learn_spell(self, video_path):
        """ STEPS
        1. read video input
        2. compute frame difference (temporal data)
        3. theshold HSV colors 
        4. clean-up image (more thresholding + closing)
        5. extract next move position from the frame
        6. scale the centers
        """
        self.video_path = video_path
        
        move_positions = self._load_known_spells()
        if len(move_positions) > 0:
            self.move_positions = move_positions
            return

        self.video = utils.read_all_frames(self.video_path, self.resolution)

        differences = utils.compute_frame_difference(self.video)
        thresholded = utils.hsv_threshold(differences, self.hsv_threshold_lower, self.hsv_threshold_upper)
        move_positions = self._get_move_positions(thresholded)
        self.move_positions = self._scale_move_positions(move_positions)
        
        self._save_knwon_spell()

    def cast_spell_in_paint(self):
        sutil = ScreenUtils()

        target_width = sutil.roi_end[0] - sutil.roi_start[0]
        target_height = sutil.roi_end[1] - sutil.roi_start[1]

        convert_point = lambda x: x * np.array([target_width, target_height]) + sutil.roi_start

        point = convert_point(self.move_positions[0])
        pyautogui.moveTo(point[0], point[1])
        pyautogui.mouseDown()
        for point in self.move_positions[1:]:
            point = convert_point(point)
            pyautogui.moveTo(point[0], point[1])
        pyautogui.mouseUp()

    def cast_spell_in_game(self, game_resolution):
        # Cursor is captured, thus all points need to be converted to relative movement
        rel_points = []
        for p1, p2 in zip(self.move_positions[:-1], self.move_positions[1:]):
            rel_points.append(p2-p1)

        convert_point = lambda x: x * np.array([game_resolution[0], game_resolution[1]]) * 1.5

        time.sleep(3)
        pyautogui.mouseDown()
        for point in rel_points:
            point = convert_point(point)
            pyautogui.moveRel(point[0], point[1])
        pyautogui.mouseUp()

    def _save_knwon_spell(self):
        video_stem = Path(self.video_path).stem
        spell_path = Path("spell_patterns") / (video_stem + ".txt")

        positions = [x.tolist() for x in self.move_positions]

        with open(spell_path, 'w') as f:
            f.write(str(positions))

    def _load_known_spells(self):
        video_stem = Path(self.video_path).stem
        spell_path = Path("spell_patterns") / (video_stem + ".txt")

        if not spell_path.exists():
            return []
        
        data = open(spell_path, 'r').read()
        return np.array(eval(data))

    def _get_move_positions(self, frames):
        # Use the debug frames to highilight detected contours and their center
        move_positions = []
        for frame, debug_frame in zip(frames, self.video):
            contour_center, debug_frame = self._extract_move_position_from_frame(frame, debug_frame.copy())
            
            # Manually check whether the detected point is ok
            if len(contour_center) > 0:
                cv2.imshow('debug frame', debug_frame)
                key = cv2.waitKey()
                if key & 0xFF == ord('q'):
                    continue
                move_positions.append(contour_center)

        cv2.destroyAllWindows()
        return move_positions

    def _scale_move_positions(self, move_positions):
        return [
            np.array([c[0] / self.resolution[0], c[1] / self.resolution[1]]) 
            for c in move_positions
        ]

    def _extract_move_position_from_frame(self, frame, debug_frame=None):
        clean_frame = self._clean_frame(frame)
        contours, h = cv2.findContours(clean_frame, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        best_cnt = None
        max_area = self.min_circle_area
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, .03 * cv2.arcLength(cnt, True), True)
            if len(approx) > 4:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
        
        contour_center = []
        if best_cnt is not None:
            M = cv2.moments(best_cnt)
            center_X = int(M["m10"] / M["m00"])
            center_Y = int(M["m01"] / M["m00"])
            contour_center = [center_X, center_Y]

            if debug_frame is not None:
                cv2.drawContours(debug_frame, [best_cnt], 0, (220, 152, 91), -1)
                cv2.circle(debug_frame, contour_center, 3, (100, 255, 0), 2)
        
        return contour_center, debug_frame

    def _clean_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,1)
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return dilation

if __name__ == "__main__":
    caster = CastingSpells()
    caster.learn_spell("videos\spell_0.mp4")
    caster.cast_spell_in_paint()