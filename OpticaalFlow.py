import numpy as np
import cv2
import time
from math import atan2,pi
start = time.time()
lk_params = dict(winSize  = (10, 10),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 20,
                    qualityLevel = 0.3,
                    minDistance = 10,
                    blockSize = 7 )


trajectory_len = 20
detect_interval = 1
trajectories = []
frame_idx = 5


cap = cv2.VideoCapture(0)


while True:
    end = time.time()
    # start time to calculate FPS
    
    
    
    
    suc, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()
    if (round(end-start))%1.5==0:
    
        # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
        if len(trajectories) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
            p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1

            new_trajectories = []

            # Get all the trajectories
            for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                trajectory.append((x, y))
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
                print("x= %d", x)
                print("y = %d" , y) 
                len(trajectory) > trajectory_len
                    
                del trajectory[0]
                new_trajectories.append(trajectory)
                # Newest detected point
                
            trajectories = new_trajectories

            # Draw all the trajectories
            cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
            cv2.putText(img, 'aci: %d' % len(trajectories), (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
            angle = np.arctan(y / x) 
            angle = (angle * 180) / np.pi 
            #print(angle)
            #print(trajectories)
            if angle > 1:
                cv2.putText(img, 'Right  :' + str(int(angle))+' degrees',
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 3, cv2.LINE_4)
            elif angle < -1:
                cv2.putText(img, 'left :' + str(int(angle))+' degrees',
                        (20,75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_4)
            else:
                cv2.putText(img, 'durgun :', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_4)
            

        # Update interval - When to update and detect new features
        if frame_idx % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            # Lastest point in latest trajectory
            for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            # Detect the good features to track
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                # If good features can be tracked - add that to the trajectories
                for x, y in np.float32(p).reshape(-1, 2):
                    trajectories.append([(x, y)])


        frame_idx += 1
        prev_gray = frame_gray

        # End time
        
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        
    # Show Results
        cv2.putText(img, f"{fps:.2f} FPS", (15, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        cv2.imshow('Mask', mask)
    cv2.imshow('Optical Flow', img)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    #print(end-start)

cap.release()
cv2.destroyAllWindows()