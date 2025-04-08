import cv2
import numpy as np
import requests
from config import IP_WEBCAM_URL
import os


# def get_video_frame():
#     try:
#         cap = cv2.VideoCapture(f"{IP_WEBCAM_URL}/video")
#         if not cap.isOpened():
#             print("⚠️ Unable to open webcam stream.")
#             return None

#         ret, frame = cap.read()
#         cap.release()

#         if ret:
#             print("📸 Frame captured successfully.")
#             return frame
#         else:
#             print("⚠️ Failed to read frame from stream.")
#             return None
#     except Exception as e:
#         print("❌ Exception in get_video_frame:", e)
#         return None
    

def get_video_frame(resize_dim=(160, 320), save_frame=True):
    try:
        cap = cv2.VideoCapture(f"{IP_WEBCAM_URL}/video")
        if not cap.isOpened():
            print("⚠️ Unable to open webcam stream.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("⚠️ Failed to read frame from stream.")
            return None

        resized = cv2.resize(frame, resize_dim)  # Resize to 160x320 (W x H)
        print(f"📸 Frame captured and resized to {resize_dim}")

        if save_frame:
            cv2.imwrite("frame.jpg", resized)
            print("💾 Frame saved as 'frame.jpg'")
           
            # if os.path.exists("frame.jpg"):
            #     print("✅ Opening frame.jpg...")
            #     cv2.imshow("Captured Frame", resized)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # else:
            #     print("❌ Image not saved.")

        return resized

    except Exception as e:
        print("❌ Exception in get_video_frame:", e)
        return None