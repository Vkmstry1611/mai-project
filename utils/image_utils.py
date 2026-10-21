from PIL import Image, ImageOps, ImageDraw

def normalize_image(image, thumbnail=True):
    if image is None:
        raise ValueError('Please upload an image or take a camera photo.')
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > 40_000_000:
        raise ValueError('Image is too large. Please resize it below 40 megapixels.')
    image = image.convert('RGB')
    if thumbnail:
        image.thumbnail((1600, 1600))
    return image

def annotate(image, evidence):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for item in evidence:
        box = item.get('box')
        if box:
            draw.rectangle(box, outline='red', width=3)
            draw.text((box[0], box[1]), item['text'], fill='red')
    return result
