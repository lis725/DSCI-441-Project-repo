Data setup instructions
=======================

Raw image data is not included in this repository or submission zip because the
Kaggle datasets are large. Download the datasets yourself and place the extracted
folders under:

  data/raw/

Datasets used or supported in the final project
-----------------------------------------------

1. Face Mask Detection ~12K Images Dataset
   Kaggle slug: ashishjangra27/face-mask-12k-images-dataset
   URL: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

2. Face Mask Detection Dataset
   Kaggle slug: omkargurav/face-mask-dataset
   URL: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset

3. Face Mask Dataset (Train-Test-Validation)
   Kaggle slug: busrabetulcavusoglu/face-mask-dataset-tvt
   URL: https://www.kaggle.com/datasets/busrabetulcavusoglu/face-mask-dataset-tvt

Recommended folder layout
-------------------------

The code is flexible, but the safest layout is:

  data/raw/
    Face Mask Dataset/
      Train/
        WithMask/
        WithoutMask/
      Validation/
        WithMask/
        WithoutMask/
      Test/
        WithMask/
        WithoutMask/
    extra_omkar/
      Train/
        WithMask/
        WithoutMask/
    extra_tvt/
      Train/
        WithMask/
        WithoutMask/
      Validation/
        WithMask/
        WithoutMask/
      Test/
        WithMask/
        WithoutMask/

Supported class folder names
----------------------------

The code recognizes these label folder names, ignoring case, spaces,
underscores, hyphens, and dots:

  WithMask, with_mask, mask, masked, wearing_mask
  WithoutMask, without_mask, no_mask, unmasked, not_mask

After placing data
------------------

Check the discovered image counts with:

  python aux_1.py --data-dir data/raw

Then train the final fixed model with:

  python main.py train --data-dir data/raw --model-path models/mask_model.joblib --image-size 128 128 --bootstrap-samples 500
