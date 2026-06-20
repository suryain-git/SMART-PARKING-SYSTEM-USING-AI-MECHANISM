import cv2
import numpy as np

VIDEO_PATH = "uploads/Aerial Shot Of Parking Lot  Stock Video.mp4"  # change this

cap = cv2.VideoCapture(VIDEO_PATH)
ret, original_frame = cap.read()
cap.release()

if not ret:
    print("Could not read video!")
    exit()

frame = original_frame.copy()
clicks = []
slots = []

def redraw():
    global frame
    frame = original_frame.copy()

    #  completed slots
    for pts in slots:
        pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

    #  current in-progress clicks
    for pt in clicks:
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
    if len(clicks) > 1:
        for i in range(len(clicks) - 1):
            cv2.line(frame, clicks[i], clicks[i+1], (0, 255, 255), 1)

    # it Show click count guide
    cv2.putText(frame, f"Clicks: {len(clicks)}/4  |  Slots marked: {len(slots)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Click 4 corners of slot | Z=Undo | S=Save | Q=Quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Pick Slots", frame)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"  Point {len(clicks)}: ({x}, {y})")

        if len(clicks) == 4:
            slots.append(clicks.copy())
            print(f"Slot {len(slots)} added with 4 points!")
            clicks.clear()

        redraw()

cv2.imshow("Pick Slots", frame)
cv2.setMouseCallback("Pick Slots", click_event)

print("Instructions:")
print("  Click 4 CORNERS of each parking slot (any order, go around the slot)")
print("  Press Z to undo last slot")
print("  Press S to save coordinates")
print("  Press Q to quit")

while True:
    key = cv2.waitKey(0)

    if key == ord('z') or key == ord('Z'):
        if clicks:
            # this line undo last single click
            removed = clicks.pop()
            print(f"Removed point: {removed}")
        elif slots:
            # this line undo last completed slot
            slots.pop()
            print(f"Removed last slot! Slots remaining: {len(slots)}")
        else:
            print("Nothing to undo!")
        redraw()

    elif key == ord('s') or key == ord('S'):
        print("\n--- COPY THIS INTO app.py ---")
        print("parking_slots = [")
        for pts in slots:
            print(f"    {pts},")
        print("]")
        break

    elif key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
