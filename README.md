# Deep Learning Models for Mammograms

### Dataset

#### MIAS
Download the dataset from this [link](http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz)

### Image Preprocess
1. Convert image from pgm format to png
   
   There are multiple ways to convert images to png. I preferred to use mogrify from ImageMagick in command line
   
   ```bash
   mogrify -format png path/to/files/*.pgm
   ```

2. Remove Noise

    In order to remove the noise from the image the biggest contour(largest blob) is found from the image. Mask of the biggest contour is used to clean the images
    
    [Code](https://github.com/ethirajsrinivasan/mammogram_deep_learning/blob/master/contour_detection/clear_image.py)
    
### Data Preparation

   Images are split into benign, malignant and normal. Train and test sets are split at 80:20 
   
   [Code](https://github.com/ethirajsrinivasan/mammogram_deep_learning/blob/master/utils/train_test_split_script.ipynb)
    
### Models

#### VGG16

   Vgg16 is used to run the classifcation model. The model is finetuned to classify malignant, benign and normal images
   
   [Code](https://github.com/ethirajsrinivasan/mammogram_deep_learning/blob/master/vgg16/vgg16_classification.ipynb)
