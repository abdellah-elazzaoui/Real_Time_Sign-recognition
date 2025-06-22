import os
import cv2

Data_dir = "./data"
if not os.path.exists(Data_dir):
    os.makedirs(Data_dir)

number_of_clusters = 3
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_clusters):
    class_dir = os.path.join(Data_dir, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for class {j}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame=cv2.resize(frame,(600,500))
        cv2.putText(frame, "If you are ready, press 'Q'", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue
        frame=cv2.resize(frame,(600,500))        
        cv2.imshow("Frame", frame)
        filename = os.path.join(class_dir, f"{counter}.jpg")
        cv2.imwrite(filename, frame)
        counter += 1
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
