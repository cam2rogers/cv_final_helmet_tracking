import os
import cv2
import numpy as np

path = "./Images"
files = os.listdir(path)

# model
model = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")

# get classes
#classes = []
#with open("./coco.names", "r") as f:
  #classes = f.read().splitlines()
# I elected to not use the coco.names file and instead only have two classes
classes = ['person']

def main():
    # loop through images in image folder
    for i in range(len(files)):
        # load image
        img = cv2.imread("./Images/sample" + str(i) + ".jpg")
        (newW, newH) = (320, 320)
        temp = cv2.resize(img, (newW, newH))
        blob = cv2.dnn.blobFromImage(temp, 1/255, (320, 320), (0, 0, 0), swapRB = True, crop = False)

        # model set up
        model.setInput(blob)
        layer_output = model.forward(['yolo_82', 'yolo_94', 'yolo_106'])

        # get boxes
        boxes = []
        confidences = []
        class_ids = []
        height, width, _ = temp.shape

        for output in layer_output:
          for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if confidence > 0.75:
              center_x = int(detection[0] * width)
              center_y = int(detection[1] * height)

              w = int(detection[2] * width)
              h = int(detection[3] * height)
 
              x = int(center_x - w/2)
              y = int(center_y - h/2)

              boxes.append([x,y,w,h])
              confidences.append(float(confidence))
              class_ids.append(class_id)

        # put boxes on image
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        color = [255, 255, 255]

        # attempt to get rid of redudant boxes
    

        for j in indexes.flatten():
          x, y, w, h = boxes[j]

          label = str(classes[class_ids[j]])
          confi = str(round(confidences[j], 2))

          cv2.rectangle(temp, (x, y), (x+w, y+h), color, 3)
          cv2.putText(temp, label + " " + confi, (x, y+20), font, 1, (255,255,255), 1)

        # save image
        cv2.imwrite("./Images/output_sample" + str(i) + ".jpg", temp)

if __name__ == '__main__':
    main()
