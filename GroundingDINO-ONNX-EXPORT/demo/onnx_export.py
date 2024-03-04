from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("/home/magica/Triton/GroundingDINO-ONNX-EXPORT/groundingdino/config/GroundingDINO_SwinB_cfg.py", "/home/magica/Triton/GroundingDINO-ONNX-EXPORT/groundingdino/config/weights/groundingdino_swinb_cogcoor.pth")
IMAGE_PATH = "/home/magica/Triton/GroundingDINO-ONNX-EXPORT/demo/cat.jpg"
TEXT_PROMPT = "head."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

model.move_op()

# 导出 onnx
model.export_onnx_enable(True)

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device="cpu"
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)