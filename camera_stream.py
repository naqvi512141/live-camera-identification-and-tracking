"""
camera_stream.py
-----------------
CameraStream: a threaded RTSP reader.

read froim the cameras continously and expose the 
mosty recent frame 

"""


import cv2
import threading
import time



class CameraStream:

    def __init__(self, name , src , reconnect_delay=5):
        self.name= name
        self.src= src
        self.reconnect_delay = reconnect_delay


        self.cap=None
        self.frame=None
        self.lock= threading.Lock()

        self.running= False
        self.thread= None


        self.frame_count=0
        self.last_frame_time= 0.0
        self.connected= False


        self._connect()

    def _connect(self):
        # reopne the RTSP connectioin

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.src)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = self.cap.isOpened()

        if self.connected:
            print(f"[{self.name}] connected.")

        else:
            print(f"[{self.name}] failed to connected.")



    def start(self):
     self.running= True
     self.thread = threading.Thread(
         target=self._update, name= f"CamThread-{self.name}", daemon=True

     )

     self.thread.start()
     return self
    

    def _update(self):
         
        """Runs in the background thread"""

        while self.running:
            if not self.connected:
                time.sleep(self.reconnect_delay)
                self._connect()
                continue

            ret , frame = self.cap.read()

            if not ret or frame is None:
                 print(f"[{self.name}] lost connection, retrying in "
                      f"{self.reconnect_delay}s...")

                 self.connected = False
                 time.sleep(self.reconnect_delay)
                 self._connect()
                 continue


            with self.lock:
                self.frame= frame
                self.frame_count +=1
                self.last_frame_time= time.time()


    def read(self):
        """Return a COPY of the latest frame, or None if nothing yet."""


        with self.lock:
            if self.frame is None :
                return None 

            return self.frame.copy()


    def is_stale( self , max_age_second=2.0):
        """True if we haven't received a new frame recently (camera likely stuck)."""

        if self.last_frame_time ==0:
            return True

        return (time.time() - self.last_frame_time) > max_age_second


    def stop(self):

        """Cleanly shut down the reader thread and release the camera."""
        self.running=False
        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.cap is not None:
            self.cap.release()


        print(f"[{self.name}] stopped.")




            

