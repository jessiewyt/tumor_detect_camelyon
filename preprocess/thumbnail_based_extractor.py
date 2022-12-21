import pandas as pd
from skimage.filters import threshold_otsu
import numpy as np
from openslide import open_slide, __library_version__ as openslide_version
import os
import glob
from preprocess.downsample import downsampler
from openslide.deepzoom import DeepZoomGenerator

from PIL import Image


"""
We use non-overlap extraction to generate small tiles from the highest resolution level,
and create a single tower model on top of that.
"""

def thumbnail_filter(slide_path, mask_path = None, tile_size = 299):
  """
  Create thumbnail from level 0 image
  Use otsu_algorithm to determine non-tissue tiles
  If there's mask, then it's posstive sample
  We use thumbnail from mask to determine cancerous tiles
  Remove non-tissue tiles

  Return a dataframe containing tiles location
  """
  with open_slide(slide_path) as slide:
    #print('dim: ', slide.dimensions[0])
    thumbnail = slide.get_thumbnail((slide.dimensions[0] / tile_size, slide.dimensions[1] / tile_size))


    thumbnail_grey = np.array(thumbnail.convert('L')) # convert to grayscale
    thresh = threshold_otsu(thumbnail_grey)
    binary = thumbnail_grey > thresh

    patches = pd.DataFrame(pd.DataFrame(binary).stack())
    patches['is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)
    patches['slide_path'] = slide_path


  with open_slide(str(mask_path)) as truth:
      thumbnail_truth = truth.get_thumbnail((truth.dimensions[0] / tile_size, truth.dimensions[1] / tile_size)) 
  
  patches_y = pd.DataFrame(pd.DataFrame(np.array(thumbnail_truth.convert("L"))).stack())
  patches_y['is_tumor'] = patches_y[0] > 0
  patches_y.drop(0, axis=1, inplace=True)

  samples = pd.concat([patches, patches_y], axis=1)
  #sampletotal.append(pd.concat([patches, patches_y], axis=1))

  samples = samples[samples.is_tissue == True] # remove patches with no tissue
  samples['tile_loc'] = list(samples.index)
  samples.reset_index(inplace=True, drop=True)
  return samples

# Mask Info for slide 038 is missing, so exclude it         
tifs = glob.glob("/content/drive/MyDrive/slides/*.tif")
masks = [item for item in tifs if 'mask' in item]
images = [item for item in tifs if not 'mask' in item and not '038' in item

#Generate tiles sample
total_samples = pd.DataFrame()
for mask in masks:
  slide_path = mask.replace('_mask', '')
  tiles = thumbnail_filter(slide_path, mask)
  total_samples = total_samples.append(tiles, ignore_index=True)


balanced_samples = downsampler(total_samples)


def generate_images(img_paths, sample_df, level = 0, tile_size = 299):

  if not os.path.exists(f'/content/drive/MyDrive/patches/level_{level}_img'):
    os.mkdir(f'/content/drive/MyDrive/patches/level_{level}_img')

  for img in img_paths:
    img_name = img.split('/')[-1]
    img_name = img_name.split('.')[0]
    print('Current processing: ', img_name)
    img_df = sample_df[sample_df['slide_path']==img]
    img_df = img_df.sample(10, replace = False, random_state=1)
    with open_slide(img) as slide:
      #
      tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
      #print('tile levels: ', tiles.level_dimensions)
      #print()
      for i in range(len(img_df)):
        cur_tile = img_df.iloc[i, :]
        tile = tiles.get_tile(tiles.level_count-level-1, cur_tile.tile_loc[::-1])
        loc_x, loc_y = cur_tile.tile_loc[::-1]
        im = np.array(tile)
        is_tumor = 'tumor' if cur_tile.is_tumor else 'normal'
        plt.imsave('/content/drive/MyDrive/patches/level_%d_img/%s_%s_%d_%d_%d.png' % (level, is_tumor, img_name, loc_x, loc_y, level), im)



    mask = img.replace('.tif', '_mask.tif')
    with open_slide(str(mask)) as truth:
      truth_tiles = DeepZoomGenerator(truth, tile_size=299, overlap=0, limit_bounds=False)
      #print('truth levels: ', truth_tiles.level_dimensions)
      #print()
      for i in range(len(img_df)):
        cur_tile = img_df.iloc[i, :]
        tile = truth_tiles.get_tile(truth_tiles.level_count-level-1, cur_tile.tile_loc[::-1])
        loc_x, loc_y = cur_tile.tile_loc[::-1]
        im = np.array(tile)
        is_tumor = 'tumor' if cur_tile.is_tumor else 'normal'
        plt.imsave('/content/drive/MyDrive/patches/level_%d_mask/%s_%s_%d_%d_%d.png' % (level, is_tumor, img_name, loc_x, loc_y, level), im)
      
generate_images(images, balanced_samples, level=0)