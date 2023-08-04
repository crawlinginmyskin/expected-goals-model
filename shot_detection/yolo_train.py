from ultralytics import YOLO

if __name__ == "__main__":
    #cap = cv2.VideoCapture('mov2.mp4')
    model = YOLO("yolov8m.pt")
    model.train(data="football_shot.yaml", epochs=50)
    model.export(format='onnx')
