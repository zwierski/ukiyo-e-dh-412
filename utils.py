
import os
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps

def resized_image(input_image):
    # Get the original width and height of the image
    width, height = input_image.size
    print("Original image size: ", width, height)

    # Calculate the aspect ratio of the image
    aspect_ratio = width / height

    # Set the desired output size
    output_size = (1000, int(1000 / aspect_ratio))

    # Resize the image while maintaining aspect ratio
    resized_image = input_image.resize(output_size)

    # Pad the resized image to create a square image
    padded_image = ImageOps.pad(resized_image, (1000, 600))

    return padded_image


def resize(url):
    try:
        path = './ukiyo-e_crawler/images/' + url.split('/')[-1][0:-4] + '.jpg'
        name = path.split('/')[-1]
        if os.path.exists(path):
            image = Image.open(path)
            r_image = resized_image(image)
            r_image = r_image.save('./ukiyo-e_crawler/resized/' + name)
        else:
            return(url, False) 
    except:
        print('error: ', url)

def download_image(url):
    try:
        path = './ukiyo-e_crawler/images/' + url.split('/')[-1][0:-4] + '.jpg'
        response = requests.get(url)
        with open(path, 'wb') as f:
            f.write(response.content)
    except:
        print('error: ', url)

def check_download(url):
    try:
        path = './ukiyo-e_crawler/images/' + url.split('/')[-1][0:-4] + '.jpg'
        if os.path.exists(path):
            return(url, True) 
        else:
            return(url, False) 
    except:
        print('error: ', url)

