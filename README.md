# Research topic: Computer vision - based detection of neurons in 2D slices of mouse brain images obtained from tissue clearing and Light-Sheet Fluoresence Microscopy

Author: Huynh Ngoc Tram Nguyen

Supervisor: Dr. Dimitri Perrin

Queensland University of Technology

# This repository includes:

- 'main.py': run this file to perform model training, (or load the model architecture and weights), do prediction, and compare human annotation with machine annotation at pixel levels.
- 'Comparison_algorithms.ipynb': a notebook file to run comparison between human annotation with machine annotation at neuron levels.
- 'Raw' folder: contains raw images.
- 'Overlays' folder: contains images with annotated points 
- 'Mask' folder: binary masks of overlay images. 
- 'Compare_label' folder: contains human annotation and machine annotation images for comparison.
- 'unet' folder: contains codes for model architecture, loss, utils - adapting from Transfer Learning DeepFlash2 (Griebel, 2021) and MIT.
- 'HTML_outputs folder: contains html files that have the outputs of main.py and Comparison_algorithms.ipynb. 

# Libraries need to be installed to run the codes:
- pandas
- numpy
- unet
- matplotlib
- keras
- tensorflow
- os
- requests
- skimage
- spicy
- imageio
- PIL 
- cvs
- ast


