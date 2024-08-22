
# Segmentation and morphological analysis of the soma location on the bright field microscopy images of neurons
Neuron soma segmentation will be performed using a bright field microscope. The first step here is to create labels from fluorescent microscopy images. While creating the label, images marked directly by biologists could also be used, but since it takes a long time and effort it was decided to create new labels from scratch. 
To create a label, the noise will be removed from the images first by using the Gaussian function. This method will be supported by the wavelet shrinkage method and Directional Ratio method will be used for neuron soma detection. In this way, detected somas will be segmented with the fast marching method. After creating the labels,
the next step of the project is to train the Unet network with the created labels and their corresponding bright field images. This network will then take bright field test images only and do soma segmentation on the image as the output. This method will pave the way for automatic cell analysis in complex bright field culture images
without damaging the cell structure.

## Creating the labels
- Soma detection
- Soma segmentation
## Creating subimages
- Preprocessing
- Soma detection
## Segmentation framework
- U-net

<p align="center">
  <img src="https://github.com/ilknurakcay/Neuron_analysis/blob/main/project_overview.png" alt="Ekran Görüntüsü" />
</p>





