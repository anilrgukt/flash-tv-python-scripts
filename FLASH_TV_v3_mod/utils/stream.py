import cv2
import threading
from datetime import datetime

class WebcamVideoStream:
    """
    Reference:
    https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """

    def __init__(self):
        self.vid = None
        self.running = False
        return

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        return

    def start(self, src, width=None, height=None, fps=None):
        # initialize the video camera stream and read the first frame
        self.vid = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.vid.isOpened():
            # camera failed
            print('THE camera could not be opened', datetime.now())
            raise IOError(("Couldn't open video file or webcam at", str(datetime.now())))
        if width is not None and height is not None:
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
        	self.vid.set(5, fps)
        self.ret, self.frame = self.vid.read()
        if not self.ret:
            self.vid.release()
            raise IOError(("Couldn't open video frame."))

        # initialize the variable used to indicate if the thread should
        # check camera vid shape
        self.real_width = int(self.vid.get(3))
        self.real_height = int(self.vid.get(4))
        self.real_fps = int(self.vid.get(5))
        print("Start video stream with shape and fps: {},{},{}".format(self.real_width, self.real_height, self.real_fps))
        self.running = True

        # start the thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.setDaemon(True)
        t.start()
        return self

    def update(self):
        try:
            # keep looping infinitely until the stream is closed
            while self.running:
                # otherwise, read the next frame from the stream
                self.ret, self.frame = self.vid.read()
        except:
            import traceback
            traceback.print_exc()
            self.running = False
        finally:
            # if the thread indicator variable is set, stop the thread
            self.vid.release()
        return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        self.running = False
        if self.vid.isOpened():
            self.vid.release()
        return

class TimeStampedFrame:
    def __init__(self, frame, timestamp):
        self.frame = frame
        self.timestamp = timestamp

class VideoQueueThread(threading.Thread):
    def __init__(self, video_path):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.queue = []
        self.stopped = False

    def run(self, fps=30, width=1920, height=1080):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            # camera failed
            print('THE camera could not be opened', datetime.now())
            raise IOError(("Couldn't open video file or webcam at", str(datetime.now())))
            
        codec = cv2.VideoWriter_fourcc('M','J','P','G')
        cap.set(6, codec)
        cap.set(5, fps) 
        cap.set(3, width)
        cap.set(4, height)
        

        while not self.stopped:
            ret, frame = cap.read()
            if ret:
                timestamp = time.time()
                self.queue.append(TimeStampedFrame(frame, timestamp))
            else:
                cap.release()
                print('Corrupt frame, could not be opened', datetime.now())
                raise IOError(("Couldn't open video frame."))
                #break
        
        cap.release()

    def stop(self):
        self.stopped = True



        
def frame_write(q, frm_count):
	idx = cam_id()
	cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
	codec = cv2.VideoWriter_fourcc('M','J','P','G')
	cap.set(6, codec)
	cap.set(5, 30)

	cap.set(3, 1920)
	cap.set(4, 1080)

	fps = int(cap.get(5))
	#print('fps: ', fps)

	count = frm_count
	write_img = True

	global stop_capture
	stop_capture = False
	
	t_st = time.time()
	timer_sec = 0
	last_frame_time = None
	fps_count = 1
	data_list = []
	
	while(cap.isOpened() and not stop_capture): 
		ret, frame = cap.read()
		frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
		time_now = datetime.now()
		
		if last_frame_time is not None:
			timer_sec = timer_sec + (frame_time - last_frame_time)
			fps_count += 1

		if not ret:
			break
			
		if timer_sec >= 1940 and timer_sec <=2060:
			write_img = not write_img
			timer_sec = 0

		
		if write_img:
			data_list.append([frame, count, time_now])
			#cv2.imwrite(os.path.join(frm_write_path, str(count).zfill(6)+'.png')
		
		if len(data_list)==7:
			a = q.put(data_list)
			write_img = not write_img
			data_list = []
			
		count += 1
		
		if (count+1)%1000 == 0:
			tmp = 10
			print('time for captuing 1000 images:::: ', time.time()-t_st)
			t_st = time.time()
			#break
		
		last_frame_time = frame_time
	
	print('The cam for batch processing is stopped.')
	cap.release()
	cv2.destroyAllWindows()

