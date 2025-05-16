import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 100, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,200), 4)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

'''def list_cameras(max_index=5):
    for i in range(1,max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[1]:
            print(f"Camera found at index {i}")
        cap.release()

list_cameras()

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())'''
