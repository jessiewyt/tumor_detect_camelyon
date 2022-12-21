import glob
import matplotlib.pyplot as plt
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
from skimage.color import rgb2gray
import pandas as pd
import os
from preprocess.downsample import downsampler
"""
We use a sliding window apporach to slide over tiles at a high resolution level, 
then find its cooresponding low resolution tile if it's a valid sample.
This extractor is used to train a two tower model.
"""

# Starter codes provided by professor!!
def read_slide(slide, x, y, level, width, height, as_float=False):
    im = slide.read_region((x,y), level, (width, height))
    im = im.convert('RGB') # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)
    return im

def find_tissue_pixels(image, intensity=0.8):
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)
    return zip(indices[0], indices[1])
  

tifs = glob.glob("/content/drive/MyDrive/slides/*.tif")
masks = [item for item in tifs if 'mask' in item]
images = [item for item in tifs if not 'mask' in item and not '038' in item]

TISSUE_THRESHOLD = 0.5
## Use level 4 and level 2 Image as input
def generate_samples(image_path, mask_path, high_lvl = 2, low_lvl = 4, window_size = 299):
    with open_slide(image_path) as slide, open_slide(mask_path) as mask:
    # Sliding over Level 2
    # Pari with Level 4
        image_pair_positive = []
        image_pair_negative = []
        for x in range(0, slide.level_dimensions[0][0], window_size*(2**high_lvl)):
            for y in range(0, slide.level_dimensions[0][0], window_size*(2**high_lvl)):
            # Calculate Paired Window Location at Lower Reso Level 
            #print(x, y)
                ratio = (2**low_lvl-2**high_lvl)//2

                # Top Left Coordinate from Low Reso
                x_low = x-window_size*ratio
                y_low = y-window_size*ratio
                # Bottom right Coordinate from Low Reso
                x_low_b = x_low+window_size*(2**low_lvl)
                y_low_r = y_low+window_size*(2**low_lvl)
                
                # Check If Window Pair is Out of bound
                if x_low<0 or y_low<0 or x_low_b>slide.level_dimensions[0][0] or y_low_r>slide.level_dimensions[0][1]:
                    continue
        
                cur_tile = read_slide(slide, 
                                    x = x, 
                                    y = y, 
                                    level = high_lvl, 
                                    width = window_size, 
                                    height = window_size, 
                                    )
                # Filter Tissue Region, threshold = 0.5
                tissue_pixels = find_tissue_pixels(cur_tile)
                tissue_ratio = len(list(tissue_pixels))/(window_size**2)
                if tissue_ratio>TISSUE_THRESHOLD:
                    # Determine Positve
                    cur_mask = read_slide(mask, 
                                    x = x, 
                                    y = y, 
                                    level = high_lvl, 
                                    width = window_size, 
                                    height = window_size, 
                                    )
                    cur_mask = cur_mask[:,:,0]
                    is_tumor = np.sum(cur_mask)>0

                    if is_tumor:
                        image_pair_positive.append([(x, y), (x_low, y_low)])
                    else:
                        image_pair_negative.append([(x, y), (x_low, y_low)])

    return image_pair_positive, image_pair_negative

total_samples = pd.DataFrame()
for mask_path in masks:
    img_path = mask_path.replace('_mask', '')
    positive_samples, negative_samples = generate_samples(img_path,
                                                        mask_path,
                                                        high_lvl=2,
                                                        low_lvl=4,
                                                        window_size=299)
    positive_high_reso = [item[0] for item in positive_samples]
    positive_low_reso = [item[1] for item in positive_samples]
    negative_high_reso = [item[0] for item in negative_samples]
    negative_low_reso = [item[1] for item in negative_samples]

    pos_df = pd.DataFrame({'slide':img_path.split('/')[-1],
                            'is_tumor': True,
                            'high_reso_loc': positive_high_reso,
                            'low_reso_loc':positive_low_reso})

    neg_df = pd.DataFrame({'slide':img_path.split('/')[-1],
                            'is_tumor': False,
                            'high_reso_loc': negative_high_reso,
                            'low_reso_loc': negative_low_reso})

    total_samples = total_samples.append(pos_df, ignore_index=True)
    total_samples = total_samples.append(neg_df, ignore_index=True)


balanced_samples = downsampler(total_samples)


def generate_images(img_paths, sample_df):
    if not os.path.exists('/content/drive/MyDrive/patches/img'):
        os.mkdir('/content/drive/MyDrive/patches/img')
    if not os.path.exists(f'/content/drive/MyDrive/patches/mask'):
        os.mkdir(f'/content/drive/MyDrive/patches/mask')

    for img in img_paths:
        mask_path = img.replace('.tif', '_mask.tif')
        with open_slide(img) as slide, open_slide(mask_path) as mask:
            file_name = img.split('/')[-1]
            img_name = file_name.split('.')[0]
            print('Current processing: ', img_name)
            img_name = img_name.replace('tumor_', '')
            img_df = sample_df[sample_df['slide']==file_name]
            #img_df = img_df.sample(10, replace = False, random_state=1)
            for i in range(len(img_df)):
                cur_tile = img_df.iloc[i, :]
                high_x, high_y = cur_tile.high_reso_loc
                low_x, low_y = cur_tile.low_reso_loc

                i_idx = high_x//(299*4)
                j_idx = high_y//(299*4)

                is_tumor = 'tumor' if cur_tile.is_tumor else 'normal'
                high_res_tile = read_slide(slide, 
                            x = high_x, 
                            y = high_y, 
                            level = 2, 
                            width = 299, 
                            height = 299, 
                            )
                low_res_tile = read_slide(slide, 
                            x = low_x, 
                            y = low_y, 
                            level = 4, 
                            width = 299, 
                            height = 299, 
                            )
        
                high_res_mask = read_slide(mask, 
                            x = high_x, 
                            y = high_y, 
                            level = 2, 
                            width = 299, 
                            height = 299, 
                            )
                low_res_mask = read_slide(mask, 
                            x = low_x, 
                            y = low_y, 
                            level = 4, 
                            width = 299, 
                            height = 299, 
                            )
                plt.imsave('/content/drive/MyDrive/patches/img/%s_%s_%d_%d_%d.png' % (is_tumor, img_name, i_idx, j_idx, 2), high_res_tile)
                plt.imsave('/content/drive/MyDrive/patches/img/%s_%s_%d_%d_%d.png' % (is_tumor, img_name, i_idx, j_idx, 4), low_res_tile)
                plt.imsave('/content/drive/MyDrive/patches/mask/%s_%s_%d_%d_%d.png' % (is_tumor, img_name, i_idx, j_idx, 2), high_res_mask)
                plt.imsave('/content/drive/MyDrive/patches/mask/%s_%s_%d_%d_%d.png' % (is_tumor, img_name, i_idx, j_idx, 4), low_res_mask)

generate_images(images, balanced_samples)
      

