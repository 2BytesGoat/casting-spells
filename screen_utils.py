import cv2
import numpy as np
import pyautogui


class ScreenUtils:
    def __init__(self):
        self.raw_image = None
        self.roi_image = None
        self.roi_start = None
        self.roi_end = None
        
        self.init_roi()


    def init_roi(self):
        self.raw_image = np.array(pyautogui.screenshot())
        self.is_selecting_roi = False
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self._region_select)

        cv2.imshow("Image", self.raw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def _region_select(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # if mousedown, store the x,y position of the mous
            self.is_selecting_roi = True
            self.roi_start = np.array([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and self.is_selecting_roi:
                # when dragging pressed, draw rectangle in image
                img_copy = self.raw_image.copy()
                cv2.rectangle(img_copy, self.roi_start, (x, y), (0,0,255), 2)
                cv2.imshow("Image", img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
                # on mouseUp, create subimage
                self.is_selecting_roi = False
                self.roi_end = np.array([x, y])
                # You can display the ROI if you show this image
                self.roi_image = self.raw_image[self.roi_start[1]:y, self.roi_start[0]:x]
                # cv2.imshow("subimg", sub_img)
                cv2.destroyAllWindows()