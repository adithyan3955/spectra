import cv2
import requests

# 🔹 Set your camera IP and port (from DroidCam or IP Webcam app)
CAMERA_IP = "192.168.18.125"
CAMERA_PORT = "4747"  # Use 8080 for IP Webcam
USE_DROIDCAM = True   # Set to False if using IP Webcam

# 🔹 Choose correct endpoint
if USE_DROIDCAM:
    CAMERA_SOURCE = f"http://{CAMERA_IP}:{CAMERA_PORT}/mjpegfeed"  # or /video
else:
    CAMERA_SOURCE = f"http://{CAMERA_IP}:{CAMERA_PORT}/video"

print("🎥 Connecting to camera stream...")

# 🔍 Check if stream is reachable
try:
    response = requests.get(CAMERA_SOURCE, timeout=5)
    if response.status_code != 200:
        print(f"❌ Stream returned status code: {response.status_code}")
        exit()
except requests.exceptions.RequestException as e:
    print(f"❌ Could not reach stream: {e}")
    print("💡 Make sure your phone and PC are on the same Wi-Fi and the app is running.")
    exit()

# 🎬 Open video stream
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("❌ ERROR: Could not open video stream.")
    exit()

print("✅ Connected! Press 'q' to quit.")

# 🧠 Load pre-trained people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ No frame received — check IP/connection.")
        break

    # Resize for speed
    frame_resized = cv2.resize(frame, (640, 480))

    # Detect people
    boxes, weights = hog.detectMultiScale(frame_resized, winStride=(8, 8))

    # Draw detection boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show output
    cv2.imshow("Spectra - People Detection", frame_resized)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()