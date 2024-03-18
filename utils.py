import cv2


def read_all_frames(video_path, frame_size=(200, 200)):
    frames = []

    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        resized_frame = cv2.resize(frame, frame_size)
        cv2.imshow('frame', resized_frame)
        frames.append(resized_frame)
    
    cap.release()
    return frames

def compute_frame_difference(frames):
    differences = []
    for frame_0, frame_1 in zip(frames[:-1], frames[1:]):
        dif = cv2.subtract(frame_1, frame_0)
        differences.append(dif)
    return differences

def hsv_threshold(frames, lower_threshold, upper_threshold):
    thresholded = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
        output = cv2.bitwise_and(frame, frame, mask=mask)
        thresholded.append(output)
    return thresholded