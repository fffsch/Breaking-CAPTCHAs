from ultralytics import YOLO

# if __name__ == "__main__":
#     model = YOLO('runs/detect/captcha_yolo_v82/weights/best.pt')

#     model.predict(
#         source = "images/test",
#         conf = 0.3,
#         save = True,
#         device = 0
#     )

model = YOLO('runs/detect/captcha_yolo_v82/weights/best.pt')

result = model.predict(
    source = "images/test/0c3dp-0.png",
    conf = 0.3,
    save = True,
    device = 0
)