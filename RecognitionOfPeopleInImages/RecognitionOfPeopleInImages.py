import cv2
import numpy as np
import os

with open("Models/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

net = cv2.dnn.readNet("Models/yolov3.weights", "Models/yolov3.cfg")

def detect_and_count_people_yolo(image_path):
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.1: 
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    count = 0
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # Desenhar retângulos ao redor das pessoas detectadas (opcional)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

    # Mostrar a imagem com as detecções (opcional)
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return count

def main(image_folder):
    total_people = 0
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(image_folder, filename)
            count = detect_and_count_people_yolo(image_path)
            total_people += count
            print(f"{filename}: {count} People Detected")

    print(f"Total number of people in all images: {total_people}")

if __name__ == "__main__":
    main("Images/")

