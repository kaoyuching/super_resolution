import cv2
from super_resolution.inference import sr_model, generate_sr_image


model = sr_model(model_path="/home/doriskao/project/super_resolution/models/RRDB_ESRGAN_x4_png_l.pth")

img = generate_sr_image(
    img_path="/home/doriskao/project/super_resolution/test_data/000002_1476784780.jpg",
    model=model,
)
cv2.imwrite(f"./test_img.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
