import io
import os
from typing import List, Optional
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from super_resolution.inference import sr_model, generate_sr_image

app = FastAPI()
model = sr_model(model_path="/home/doriskao/project/super_resolution/models/RRDB_ESRGAN_x4_png_l.pth")
root_path = "./output/"

@app.post("/single_img/")
async def single_sr_image(file: UploadFile):
    r"""
    Single image super resolution
    """
    file_path = io.BytesIO(file.file.read())
    img = generate_sr_image(img_path=file_path, model=model)
    res, im_png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return Response(content=im_png.tobytes(), media_type="image/png")


def show_html(imgs):
    img_tags = ""
    for img in imgs:
        img_tag = f"""
        <div>
          <img src="data:image/png;base64, {img}" alt="Red dot" />
        </div>
        """
        img_tags = img_tags + img_tag
    html_content = f"""
    <html>
        <head>
            <title>SR images</title>
        </head>
        <body>
            {img_tags}
        </body>
    </html>
    """
    return html_content


@app.post("/multi_imgs/show", response_class=HTMLResponse)
async def show_multi_sr_images(files: List[UploadFile]):
    r"""
    Multiple images super resolution
    """
    imgs = {}
    for file in files:
        file_path = io.BytesIO(file.file.read())
        img = generate_sr_image(img_path=file_path, model=model)
        res, im_png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        imgs[file.filename] = base64.b64encode(im_png).decode('utf-8')
    result = show_html(imgs.values())
    return HTMLResponse(content=result, status_code=200)


class MultiImgItem(BaseModel):
    image_id: List[str]
    image_path: List[str]


@app.post("/multi_imgs/save", response_model=MultiImgItem)
async def save_multi_sr_images(files: List[UploadFile]):
    r"""
    Multiple images super resolution
    """
    if not os.path.exists(root_path):
        os.mkdir('./output/')
    image_id = []
    image_path = []
    for file in files:
        filename = file.filename.split('.')[0]
        file_path = io.BytesIO(file.file.read())
        ofile_path = os.path.join(root_path, filename + '.png')
        img = generate_sr_image(img_path=file_path, model=model)
        cv2.imwrite(ofile_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        image_id.append(filename)
        image_path.append(os.path.abspath(ofile_path))
    return {'image_id': image_id, 'image_path': image_path}


class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


@app.post("/items/")
async def create_item(item: Item):
    return item


@app.post("/login/")
async def login(username: str = Form(...), password: str = Form(...)):
    return {"username": username}
