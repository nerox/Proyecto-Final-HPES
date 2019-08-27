import cv2 

def gstreamer_pipeline (capture_width=3840, capture_height=2160, display_width=620, display_height=480, framerate=21, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))



key = cv2. waitKey(1)
webcam = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
#webcam = cv2.VideoCapture(0)
while True:
     
    check, frame = webcam.read()
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): 
        cv2.imwrite(filename='saved_img.jpg', img=frame)
        webcam.release()
        img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_COLOR)
        img_new = cv2.imshow("Captured Image", img_new)
        cv2.waitKey(1650)
        cv2.destroyAllWindows()
     
        break
    elif key == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
