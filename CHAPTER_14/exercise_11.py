#!/usr/bin/env python3

'''
    Go through TensorFlow's Style Transfer tutorial. It is a fun way to generate art using Deep Learning.

    Design goal:
        
        Compose some image into the style of another image 
'''

import functools
import os 
from matplotlib import gridspec
import matplotlib.pyplot as plt
import tensorflow_hub as hub 
import tensorflow as tf 

CONTENT_IMAGE_URL = 'https://as2.ftcdn.net/jpg/07/89/12/31/1000_F_789123117_s0Prg8ZmWBZkpzgR5ijwypI51DFLoKdx.jpg' 
STYLE_IMAGE_URL =  'https://as1.ftcdn.net/v2/jpg/08/80/56/64/1000_F_880566413_DlJKirayA1Hr3Lsi589hcOcRaoDorTFM.jpg'
HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

def crop_center(image):
    """
        Returns a cropped square image
    """
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, new_shape, new_shape)
    
    return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
    """
        Load/preprocess images
    """
    # cache file locally 
    basename = os.path.basename(image_url)[-128:]
    cached_image_path = tf.keras.utils.get_file(fname=basename, origin=image_url)

    # load image
    raw = tf.io.read_file(cached_image_path)
    img = tf.io.decode_image(raw, channels=3, dtype=tf.float32)
    img_single_batch = img[tf.newaxis, ...]
    img_single_batch = crop_center(img_single_batch)
    img_single_batch = tf.image.resize(img_single_batch, image_size, preserve_aspect_ratio=preserve_aspect_ratio)

    return img_single_batch

def show_n (images, titles = ('',), save_figure=False):
    """
        Display images 
    """
    n = len(images)
    images_sizes = [image.shape[1] for image in images]
    
    w = (images_sizes[0]*6) //320
    figure_container_width = w * n # inches
    figure_container_height = w    # inches 

    gs = gridspec.GridSpec(1, n, width_ratios=images_sizes)
    plt.figure(figsize=(figure_container_width, figure_container_height))
    for i in range(n):
        plt.subplot(gs[i])
        plt.imshow(images[i][0], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    
    if save_figure:
        plt.savefig(os.path.join(os.getcwd(), 'exercise_11_StyleTransferExample.png'))
    else:
        plt.show()

print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

content_image = load_image(CONTENT_IMAGE_URL)
style_image = load_image(STYLE_IMAGE_URL)
style_image_avg = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

hub_module = hub.load(HUB_HANDLE) # import TF Hub module for image stylization 
outputs = hub_module(content_image, style_image_avg)
stylized_image = outputs[0]
show_n([content_image, style_image_avg, stylized_image], titles=['Original content image', 'Style image' , 'Stylized image'], save_figure=False)