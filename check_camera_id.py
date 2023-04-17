import cv2

# checks the available open camera ids
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(
            f"Camera {i}: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    cap.release()
