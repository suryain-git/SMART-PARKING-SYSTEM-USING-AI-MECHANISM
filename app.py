from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import numpy as np
from ultralytics import YOLO
import os
import datetime
import cv2
cv2.setNumThreads(1) 
import threading
cap_lock = threading.Lock()

app = Flask(__name__)   

# ---------------- UPLOAD SETTINGS ----------------
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
model = YOLO("yolov8m.pt")   # medium model — much better on aerial/top-view
model.fuse()

# Open webcam (0 = laptop camera)
cap = None
video_path = ""

# Parking slot coordinates — (top-left, bottom-right)
parking_slots = [
    [(365, 268), (404, 266), (416, 332), (371, 332)],
    [(404, 268), (440, 266), (458, 331), (416, 331)],
    [(439, 267), (477, 266), (499, 331), (459, 331)],
    [(477, 265), (511, 265), (541, 328), (498, 330)],
    [(514, 263), (550, 262), (583, 332), (541, 330)],
    [(552, 264), (587, 262), (626, 328), (581, 331)],
    [(261, 268), (295, 268), (295, 334), (254, 335)],
    [(190, 267), (225, 269), (215, 337), (175, 332)],
    [(154, 269), (191, 267), (175, 334), (134, 334)],
    [(88, 265), (122, 267), (94, 332), (55, 332)],
    [(49, 264), (15, 264), (46, 212), (77, 211)],
    [(403, 265), (393, 213), (355, 214), (365, 266)],
    [(426, 210), (459, 210), (476, 265), (439, 266)],
    [(491, 210), (523, 210), (551, 262), (513, 262)],
    [(559, 210), (587, 258), (551, 261), (524, 208)],
    [(265, 216), (297, 218), (299, 269), (261, 267)],
    [(227, 126), (248, 126), (245, 158), (217, 159)],
    [(299, 126), (327, 126), (328, 158), (300, 159)],
    [(274, 126), (299, 124), (301, 160), (274, 160)],
    [(348, 123), (375, 124), (383, 156), (352, 158)],
    [(376, 122), (402, 121), (411, 155), (383, 155)],
    [(404, 122), (429, 120), (439, 153), (410, 154)],
    [(429, 120), (453, 120), (468, 154), (438, 153)],
    [(454, 120), (482, 120), (497, 152), (468, 154)],
    [(537, 120), (560, 119), (582, 148), (554, 152)],
    [(591, 118), (610, 116), (639, 148), (615, 150)],
    [(155, 266), (189, 266), (204, 215), (172, 214)],
    [(109, 212), (130, 213), (121, 262), (86, 263)],
    [(347, 96), (342, 75), (365, 74), (371, 94)],
    [(367, 74), (387, 72), (395, 94), (369, 95)],
    [(387, 72), (409, 72), (419, 94), (394, 93)],
    [(434, 72), (455, 70), (465, 92), (441, 92)],
    [(525, 66), (545, 64), (562, 89), (538, 90)],
    [(301, 96), (301, 75), (325, 75), (325, 96)],
    [(297, 74), (283, 73), (278, 96), (300, 96)],
    [(135, 35), (153, 34), (145, 50), (126, 51)],
    [(247, 33), (266, 33), (265, 50), (245, 52)],
    [(280, 34), (302, 33), (302, 50), (279, 50)],
    [(360, 31), (375, 31), (381, 43), (361, 46)],
    [(374, 30), (397, 30), (401, 44), (379, 45)],
    [(414, 28), (396, 28), (401, 44), (421, 42)],
    [(414, 27), (432, 27), (440, 45), (420, 46)],
    [(475, 26), (492, 24), (504, 39), (482, 42)],
    [(509, 22), (494, 23), (503, 40), (525, 38)],
    [(529, 22), (511, 22), (523, 40), (543, 35)],
    [(358, 28), (355, 16), (370, 13), (373, 31)],
    [(388, 14), (393, 27), (375, 29), (369, 13)],
    [(426, 12), (407, 12), (413, 23), (433, 24)],
    [(484, 7), (489, 20), (504, 17), (501, 9)],
    [(508, 19), (527, 18), (515, 5), (501, 8)],
]

total_slots = len(parking_slots)
occupied_slots = 0
free_slots = total_slots

# Smoothing: track last N frame states per slot to avoid flickering
from collections import deque
SMOOTH_FRAMES = 5   
slot_history = [deque(maxlen=SMOOTH_FRAMES) for _ in parking_slots]   

import cv2

def get_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU) between two boxes.
    box format: (x1, y1, x2, y2)
    Returns a float between 0.0 and 1.0
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union

import cv2



VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorcycle"}
IOU_THRESHOLD = 0.2   # 20% overlap = occupied

VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorcycle"}
IOU_THRESHOLD = 0.3
BASE_W, BASE_H = 1280, 720   

def generate_frames():
    global video_path, cap
    global occupied_slots, free_slots, slot_history

    if cap is None and video_path:
        cap = cv2.VideoCapture(video_path)

    if cap is None:
        return

    while True:                                          
        with cap_lock:
            success, frame = cap.read()
        if not success:
            break
        print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")
        h, w = frame.shape[:2]                          

        ref_w = 1280                                   
        ref_h = 720                                     

        # STEP 0: Scale slots
        frame_h, frame_w = frame.shape[:2]              
        scale_x = frame_w / BASE_W                      
        scale_y = frame_h / BASE_H                      
        scaled_slots = []                               

        # No scaling needed - using exact coordinates
        scaled_slots = parking_slots

        # STEP A: YOLO detection
        results = model(frame, verbose=False, conf=0.45, iou=0.5)[0]  
        vehicle_boxes = []                              

        for box in results.boxes:                       
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            if label in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_boxes.append((x1, y1, x2, y2))

        
        # STEP B: Check each slot against detected vehicles
        
        slot_states = []
        slot_vehicle_map = {}
        for i, pts in enumerate(parking_slots):
            pts_array = np.array(pts, np.int32)
            x, y, w, h = cv2.boundingRect(pts_array)
            slot_box = (x, y, x + w, y + h)
            this_slot_occupied = False
            for vbox in vehicle_boxes:
                iou = get_iou(slot_box, vbox)
                ix1 = max(slot_box[0], vbox[0])
                iy1 = max(slot_box[1], vbox[1])
                ix2 = min(slot_box[2], vbox[2])
                iy2 = min(slot_box[3], vbox[3])
                overlap_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if iou >= IOU_THRESHOLD or overlap_area > 200:
                    this_slot_occupied = True
                    slot_vehicle_map[i] = vbox
                    break
            slot_states.append(this_slot_occupied)
        print(f"Vehicle boxes: {vehicle_boxes[:3]}")   # shows first 3 vehicle coords
        print(f"Slot 0 box: {parking_slots[0]}")        # shows first slot coords
        print(f"Matched slots: {len(slot_vehicle_map)}")
        # STEP C: Smoothing
        smoothed_states = []                            
        for i, current_state in enumerate(slot_states):
            slot_history[i].append(current_state)
            votes_occupied = sum(slot_history[i])
            smoothed = votes_occupied >= (len(slot_history[i]) * 0.6)
            smoothed_states.append(smoothed)

        # STEP D: Update counters
        occupied_slots = sum(smoothed_states)           
        free_slots = total_slots - occupied_slots       

        # STEP E: Draw parking slots
        for i, pts in enumerate(parking_slots):
            is_occupied = smoothed_states[i]
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_array], isClosed=True, color=color, thickness=2)
            if slot_states[i] and i in slot_vehicle_map:
                vx1, vy1, vx2, vy2 = slot_vehicle_map[i]
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)    

        # STEP F: Summary overlay
        overlay_text = [                                
            f"Total  : {total_slots}",
            f"Free   : {free_slots}",
            f"Occup  : {occupied_slots}"
        ]
        for j, line in enumerate(overlay_text):
            cv2.putText(frame, line, (10, 25 + j * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Timestamp
        current_time = datetime.datetime.now().strftime("%H:%M:%S")  
        cv2.putText(frame, current_time, (frame_w - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Legend
        cv2.putText(frame, "Green = Free", (10, frame_h - 40),  
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Red = Occupied", (10, frame_h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # STEP G: Encode and yield
        ret, buffer = cv2.imencode('.jpg', frame)       
        if not ret:                                     
            continue                                    
        yield (b'--frame\r\n'                           
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/how")
def how():
    return render_template("how.html")


@app.route("/team")
def team():
    return render_template("team.html")


from flask import Response, stream_with_context


@app.route("/stats")
def stats():
    global total_slots, free_slots, occupied_slots   

    return jsonify({
        "total": total_slots,
        "free": free_slots,
        "occupied": occupied_slots
    })

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path, cap 

    if "video" not in request.files:
        return redirect(url_for("live"))

    file = request.files['video']

    if file.filename == "":
        return redirect(url_for("live"))

    path = "uploads/" + file.filename
    file.save(path)

    video_path = path   # stores video path
    if cap:
        cap.release()
        cap = cv2.VideoCapture(video_path)
    return redirect(url_for("live"))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/webcam")
def webcam():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return redirect(url_for("live"))       

from flask import request

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global cap

    mode = request.form.get('mode')

    if cap:
        cap.release()

    if mode == "webcam":
        cap = cv2.VideoCapture(0)

    elif mode == "mobile":
        mobile_ip = request.form.get('mobile_ip')
        cap = cv2.VideoCapture(f"http://{mobile_ip}/video")

    return "Mode switched successfully"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
