from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
from imutils.video import VideoStream
import argparse
import time

#Import the predicting model
from model_predict import *

class DermAIgui:
    def __init__(self, vs, outputPath):
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.root = tki.Tk()
        self.panel = None
        btn = tki.Button(self.root, text="Capture",command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        self.root.wm_title("Derm AI")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        
    def videoLoop(self):
        try:
            while not self.stopEvent.is_set():
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=700)
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                    
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")
            
    def takeSnapshot(self):
        ts = datetime.datetime.now()
        filename = "output.jpg"
        p = os.path.sep.join((self.outputPath, filename))
        image = self.frame.copy()
        cv2.imwrite(p, self.frame.copy())
        
        print("[INFO] saved {}".format(filename))
        self.frame = image
        self.openNewWindow()
    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.destroy()
    
    def retake(self):
        newWindow.destroy()
        
    def predict(self):
        result_array = predict()
        for i in range(5):
            pop_label = tki.Label(newWindow,text=result_array[i][0]+"   "+result_array[i][1]) 
            pop_label.pack()
        
    def openNewWindow(self):
        global newWindow
        newWindow = tki.Toplevel(self.root)
        newWindow.title("DermAI-Predict")
        
        global sample_image
        global output_pic
        pop_label = tki.Label(newWindow,text="Predict??") 
        pop_label.pack(pady=10)
        sample_image = ImageTk.PhotoImage(Image.open('/home/pi/Pictures/output.jpg'))
         
        new_frame = tki.Frame(newWindow)
        new_frame.pack(pady=5)
        
        output_pic=tki.Label(new_frame,image=sample_image,borderwidth=0)
        output_pic.pack(padx=10)
        
        btn_predict = tki.Button(new_frame,text="Predict",command=lambda: self.predict())
        btn_predict.pack(padx=10,pady=10,expand='yes',fill='both')
        
        btn_retake = tki.Button(new_frame,text="Retake",command=lambda: self.retake())
        btn_retake.pack(padx=10,pady=10,expand='yes',fill='both')
        
# construct the argument parse and parse the arguments
# initialize the video stream and allow the camera sensor to warmup
output='/home/pi/Pictures'
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
# start the app
pba = DermAIgui(vs, output)
pba.root.mainloop()
