import sys,os
import select,subprocess
import time,pdb
import numpy as np
import cv2
from matplotlib import pyplot as plt

        
def capture_both(dt=.1):
    io = os.dup(sys.stdin.fileno())
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frames = []
    samples = []

    try:
        while True:
            good,frame = cap.read()
            if good:
                frames.append(frame)
            else:
                print("Failed to read %d"%(len(frames)))

            s = subprocess.getoutput('/bin/cat /proc/net/wireless')
            rssi = int(s.split('\n')[2].split()[2].replace('.',''))
            samp_time = time.time() - start_time
            samples.append( (len(samples),samp_time,rssi) )

            time.sleep(dt)
    except KeyboardInterrupt:
        pass

    ms = np.array(samples)
    for i in range(len(frames)):
        cv2.imwrite('./images/frame_%06d.png'%i, frames[i])
        np.save('./images/measurements', ms)


def do_playback(compute_trackers=False):
    dire = './images'
    dt = .03
    frame_names = sorted([f for f in os.listdir(dire) if 'frame' in f])
    num_files = int(frame_names[-1].split('.')[0].split('_')[1])
    ms = np.load(dire + '/measurements.npy')

    if compute_trackers:
        #fastf = cv2.FastFeatureDetector_create()
        fastf = cv2.ORB_create()

    f,axs = plt.subplots(2, gridspec_kw={"height_ratios":[3,1]}, figsize=(9,7))
    plt.tight_layout()
    axs[1].plot(range(num_files+1), ms[:,2])
    yl = [0, max(ms[:,2])]
    time_line = axs[1].plot([0,0],yl, color='k')[0]
    for ii,fname in enumerate(frame_names):
        frame = cv2.imread(dire+'/'+fname)
        if ii == 0:
            plt_img = axs[0].imshow(frame)
            plt.show(block=False)
        else:
            time_line.set_data([ii,ii],yl)
            if compute_trackers:
                kp = fastf.detect(frame,None)
                cv2.drawKeypoints(frame, kp, frame, color=(255,0,30))

            plt_img.set_data(frame)
            plt.pause(dt)
        plt.title( ms[ii][2] )
        


def do_plot_samples(samps):
    plt.plot(samps[:,0], samps[:,1])
    plt.title("RRSI vs time")
    plt.show()


if 'capture' in sys.argv:
    capture_both()
elif 'playback' in sys.argv:
    do_playback(True)
else:
    print("Usage:\n\t./capture capture\n OR\n\t./capture playback")
