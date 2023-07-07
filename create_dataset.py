import glob
import os
import random
import shutil
import zipfile
from pathlib import Path

import Augmentor
import numpy as np
import requests
import re

import wget
from PIL import ImageFilter, ImageOps, ImageDraw, ImageFont
from PIL.Image import Image
from tqdm import tqdm

from multiprocessing import Pool

FONTS_FOLDER = Path('./datasets/fonts')
DATASET_FOLDER = Path('./datasets/v5')

IMAGES_PER_FONT = 500
IMAGE_SIZE = (256, 82)



def get_fonts_from_page(index):
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.143 Safari/537.36'
    headers = {'User-Agent': user_agent}
    response = requests.get(f'https://www.dafont.com/theme.php?cat=603&page={index}&fpp=200&l[]=10&l[]=1&l[]=6', headers=headers)

    return re.findall(r'<strong>(.+?)</strong></a>.*?K\" href=\"(.+?)\"  rel=\"nofollow\">&nbsp;Download&nbsp;</a>', response.text)


def get_fonts():
    fonts = []
    for i in range(1, 58):
        fonts.extend(get_fonts_from_page(i))

    return fonts


all_fonts = get_fonts()
print(f'found {len(all_fonts)} fonts.')

def download_font(font):
    to_download_path = FONTS_FOLDER

    if not os.path.isdir(FONTS_FOLDER):
        os.makedirs(FONTS_FOLDER)
    zip_path = FONTS_FOLDER / f'{font[0]}.zip'
    wget.download('https://' + font[1], zip_path)

    if os.path.isfile(zip_path):
#         print(zip_path)
        with zipfile.ZipFile(zip_path) as zip:
            zip.extractall(to_download_path)

        os.remove(zip_path)
    else:
        print(f'WARNING {zip_path} does not exist.')


pool = Pool(processes=8)

for font in tqdm(all_fonts):
    try:
        pool.apply_async(download_font, args=(font,))
        # download_font(font)
    except Exception as e:
        print(f'Failed on font: {font}', e)

pool.close()
pool.join()



word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS = response.content.splitlines()
WORDS = [word.decode("utf-8") for word in WORDS if len(word) >= 2]

bgs = [np.asarray(Image.open(bg)) for bg in glob.glob("./datasets/bg/*")]



TEMP_LOCATION = Path('./temp')
TEMP_LOCATION.mkdir(exist_ok=True)

def get_size_of_font(font_path):
    unicode_text = u"hello"
    size = int(IMAGE_SIZE[1])
    max_height = int(IMAGE_SIZE[1] * .8)
    min_height = int(IMAGE_SIZE[1] * .7)
    max_width = int(IMAGE_SIZE[0] * .8)
    min_width = int(IMAGE_SIZE[0] * .5)
    font_truetype = ImageFont.truetype(font_path, size, encoding="unic")
    left, top, right, bottom = font_truetype.getbbox(unicode_text)
    text_height = bottom - top
    text_width = right - left
    count = 0

    while not (min_height < text_height < max_height) and (min_width < text_width < max_width) and count < 400:
        count += 1

        if text_height >= 65:
            size -= 1
        else:
            size += 1

        font_truetype = ImageFont.truetype(font_path, size, encoding="unic")
        left, top, right, bottom = font_truetype.getbbox(unicode_text)
        text_height = bottom - top
        text_width = right - left

    return size, count < 400

def create_base_image(font_file: str, selected_word: str, font_size: int, output_name: str):
    font_truetype = ImageFont.truetype(font_file, random.randint(int(font_size * 0.65), int(font_size * 0.85)), encoding="unic")
    left, top, right, bottom = font_truetype.getbbox(selected_word)
    text_width = right - left
    canvas = Image.new('L', IMAGE_SIZE, "white")

    # draw the text onto the text canvas, and use black as the text color
    draw = ImageDraw.Draw(canvas)
    draw.fontmode = "L"

    if text_width > 230:
        right_margin = random.randint(0, 20)
    else:
        right_margin = random.randint(int(((300 - text_width) / 2) * .8), int(((300 - text_width) / 2) * 1.2))

    draw.text((right_margin,random.randint(-5, 25)), selected_word, 'black', font_truetype)
    canvas.save(TEMP_LOCATION / f"{output_name}.png", "PNG")
    print(TEMP_LOCATION / f"{output_name}.png")

def get_random_background(bgs: any):
    random_bg_array = random.choice(bgs)
    random_bg_image = Image.fromarray(np.uint8(random_bg_array))

    scaling_bg_w, scaling_bg_h = int(IMAGE_SIZE[0] * 1.5), int(IMAGE_SIZE[1] * 1.5)

    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    bg_width, bg_height = random_bg_image.size

    # Setting the points for cropped image
    bg_left = random.randint(5, bg_width - scaling_bg_w - 5)
    bg_top = random.randint(5, bg_height - scaling_bg_h - 5)
    bg_right = bg_left + scaling_bg_w
    bg_bottom = bg_top + scaling_bg_h

    # Cropped image of above dimension
    # (It will not change original image)
    background_cropped = ImageOps.grayscale(random_bg_image.crop((bg_left, bg_top, bg_right, bg_bottom)))
    background_sized = background_cropped.resize(IMAGE_SIZE)

    return background_sized


def temp_image_to_real(image_name: str, folder_path, image_distorted: Path, background):
    word_image = Image.open(image_distorted).convert('L')
    word_image_array = np.asarray(word_image)
    word_image.close()

    if random.random() > .666:
        word_image_array = (1 - (word_image_array / 255)) * np.random.normal(random.gauss(random.randint(130, 250), random.randint(15, 35)), random.gauss(random.randint(15, 40), random.randint(2, 6)), word_image_array.shape)
        word_image_array = Image.fromarray(np.uint8(word_image_array))
        word_image_array = word_image_array.filter(ImageFilter.SMOOTH)
        word_image_array = np.asarray(word_image_array)
        word_image_array = np.clip(word_image_array, 0, 255)

        background_sized_array = np.asarray(background)
        Image.fromarray(np.uint8(np.clip(background_sized_array - word_image_array, 0, 255))).save(folder_path / f'{image_name}.png')
    elif random.random() > .5:
        Image.fromarray(np.uint8(word_image_array * (random.random() / 5 + .8))).save(folder_path / f'{image_name}.png')
    else:
        background_sized_array = np.asarray(background)
        Image.fromarray(np.uint8(background_sized_array - (255 - word_image_array))).save(folder_path / f'{image_name}.png')

    os.remove(image_distorted)


from Augmentor.Pipeline import Operation
class RealImage(Operation):
    def __init__(self, probability, bgs):
        Operation.__init__(self, probability)
        self._bgs = bgs

    def perform_operation(self, image):
        image = image[0]
        image_array = np.asarray(image).astype(np.float32)
        random_bg_image = get_random_background(self._bgs)

        if random.random() > .5:
            word_image_array = (1 - (image_array / 255)) * np.random.normal(
                random.gauss(random.randint(130, 250), random.randint(15, 35)),
                max(random.gauss(random.randint(15, 40), random.randint(2, 6)), 0.1),
                image_array.shape
            )
            word_image_array = Image.fromarray(np.uint8(word_image_array))
            word_image_array = word_image_array.filter(ImageFilter.SMOOTH)
            word_image_array = np.asarray(word_image_array)
            word_image_array = np.clip(word_image_array, 0, 255)

            background_sized_array = np.asarray(random_bg_image)
            image_array = np.uint8(np.clip(background_sized_array - word_image_array, 0, 255))
        elif random.random() > .7:
            image_array = np.uint8(image_array * (random.random() / 5 + .8))
        else:
            background_sized_array = np.asarray(random_bg_image)
            image_array = np.uint8(np.clip(background_sized_array - (255 - image_array), 0, 255))

        image = Image.fromarray(image_array)
        return [image]


shutil.rmtree(TEMP_LOCATION)
TEMP_LOCATION.mkdir(exist_ok=True)

fonts_files = glob.glob("/datasets/fonts/*")

for fonts_file in tqdm(fonts_files):
    try:
        name = os.path.splitext(fonts_file)[0].split('/')[-1].strip()
        extension = os.path.splitext(fonts_file)[1][1:].strip()

        if extension not in ['otf', 'ttf']:
            continue

        im_path = DATASET_FOLDER / name

        try:
            os.mkdir(im_path)
        except Exception as error:
            print(error)
            continue

        size, isGood = get_size_of_font(fonts_file)

        if not isGood:
            # os.rmdir(im_path)
            continue

        for i in range(IMAGES_PER_FONT):
            font = ImageFont.truetype(fonts_file, random.randint(int(size * 0.7), int(size * 0.85)), encoding="unic")
            word = random.choice(WORDS)
            create_base_image(font_file=fonts_file, font_size=size, selected_word=word, output_name=str(i))

        p = Augmentor.Pipeline(source_directory=TEMP_LOCATION, output_directory=im_path)
        p.random_distortion(probability=.9, grid_width=5, grid_height=4, magnitude=3)
        p.rotate(probability=0.3, max_left_rotation=6, max_right_rotation=6)
        p.skew(probability=.4, magnitude=0.1)

        p.greyscale(probability=1)
        p.add_operation(RealImage(probability=1, bgs=bgs))

        p.process()

        shutil.rmtree(TEMP_LOCATION)
        TEMP_LOCATION.mkdir(exist_ok=True)
    except Exception as e:
        print(e, fonts_file)
