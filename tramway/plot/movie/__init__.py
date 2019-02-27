
import os

try:
    import cv2
except:
    try: # Windows
        os.startfile
    except:
        class Video(object):
            def __init__(self, *args, **kwargs):
                import cv2 # fails
    else:
        class Video(object):
            def __init__(self, filepath, *args, **kwargs):
                self.path = filepath
            def play(self):
                os.startfile(self.path)
else:
    # mostly borrowed from:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html#playing-video-from-file
    class Video(object):
        def __init__(self, filepath, fps=20, **kwargs):
            self.frame_duration = int(1000./float(fps))
            self.path = filepath
            self.cap = cv2.VideoCapture(self.path)
        def play(self):
            if not self.cap.isOpened():
                self.cap.open(self.path)
                if not self.cap.isOpened():
                    raise RuntimeError('cannot open the movie')
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok: break
                cv2.imshow('frame', frame)
                # wait and listen key events
                key = cv2.waitKey(self.frame_duration)
                if key & 0xFF == ord('q'): # quit on 'q'
                    break
                elif key == 27: # quit on ESC
                    break
                # check that window is still opened
                if cv2.getWindowProperty('frame', cv2.WND_PROP_AUTOSIZE) < 1:
                    break
            self.cap.release()
            cv2.destroyAllWindows()

__all__ = [ 'Video' ]
