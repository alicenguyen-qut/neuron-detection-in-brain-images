# HUYNH NGOC TRAM NGUYEN (N10596283)
# RUN THIS MAIN.PY FILE TO DO TRAINING, OR LOADING THE MODEL AND COMPARE AT PIXEL LEVELS.
# Reference: Transfer Learning model DeepFlash2 (Griebel, 2021)

#################################
# 1. Import necessary libraries##
#################################
import pandas as pd 
import numpy as np
from unet import utils
from unet import sim_measures
from unet.model import unet_1024
from unet.metrics import recall, precision, f1, mcor
from unet.losses import weighted_bce_dice_loss
import os
from matplotlib import pyplot as plt
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from tensorflow import keras


#############################
## 2. LOAD AND THE DATA #####
#############################

# load the data
img_path='Raw/'
mask_path='Mask/'
file_ids = [x.rsplit('.',1)[0] for x in os.listdir(img_path)]
file_ids.remove("")
file_names=[(os.path.join(img_path, x)) for x in [s + '.tif' for s in file_ids]]
mask_names=[(os.path.join(mask_path, x)) for x in [s + '-mask.tif' for s in file_ids]]

# prepare data for training 
img_list=file_list = [utils.readImg(x, path='', channels=1, dimensions=1024) for x in file_names]
msk_list=mask_list = [utils.readImg(x, path='', channels=1, dimensions=1024) for x in mask_names]

# for visualization - plot image sand masks
utils.plot_image_and_mask(img_names = file_names, img_list = file_list,msk_names = mask_names, msk_list = mask_list)


#############################
##3. UNET MODEL TRAINING#####
#############################

#Train model
#using Transfer Learning model: https://matjesg.github.io/deepflash2/model_library.html (with cFOS_Wue weights)
Model_name = 'cFOS_Wue'
model = utils.load_unet(Model_name)

epochs = 50
train_generator = utils.create_generator(img_list, msk_list)
model.fit_generator(train_generator,
                    steps_per_epoch=int(np.ceil(len(img_list)/4.)),
                    epochs=epochs)

# #save model if needed
# model_json = model.to_json()
# with open("model_50.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_50.h5")
# print("Saved model to disk")

# #4. Load the model if already training the model
# #load json and create model
# json_file = open('saved_model/model_50.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights("saved_model/model_50.h5")
# print("Loaded model from disk")



#############################
## 4. MODEL PREDICTION#######
#############################
pred_train = model.predict(np.asarray(img_list))
pred_train_list = [pred_train[i] for i in range(pred_train.shape[0])]



##################################
## 5. COMPARISON AT PIXEL LEVELS##
##################################

#Calculate Jaccard Similary. The computation of the ROI-wise Jaccard Similarity may take up more time.
pixelwise = True #@param {type:"boolean"}
roiwise = False #@param {type:"boolean"}

# The threshold converts the probabilistic model output to a binary segmentation mask. For example, a threshold of 0.5 means that all pixels with a 50% posiitve class probability or more are cosindered to belong to the positive class. 
threshold = 0.5 #@param {type:"slider", min:0, max:1, step:0.01}
# The minimum size (amount pixel) to be cosidered as ROI. Only applies to ROI-wise similarity.
min_roi_size = 15 #@param {type:"slider", min:1, max:250, step:1}

if pixelwise:
  jac_pix = [sim_measures.jaccard_pixelwise(a, b, threshold=threshold) for a,b 
             in zip(pred_train_list, msk_list)]

if roiwise:
  jac_roi = [sim_measures.jaccard_roiwise(a, b, threshold=threshold, min_roi_pixel=min_roi_size)
             for a,b in zip(pred_train_list, msk_list)]
 
if pixelwise and not roiwise:
  jac_str = [str('pixelwise %.2f' %pix) for pix in jac_pix]
  
if roiwise and not pixelwise: 
  jac_str = [str('ROI-wise %.2f' %roi) for roi in jac_roi]

if roiwise and pixelwise:
  jac_str = [str('pixelwise %.2f, ROI-wise: %.2f' %(pix,roi)) 
     for pix,roi in zip(jac_pix, jac_roi)]
    
_ = [print('Jaccard %s: %s' %(name,s)) for name,s in zip(file_names, jac_str)]




###################################################
## 5. VISUALISATION OF COMPARISON AT PIXEL LEVELS##
###################################################

print("Annotations:")
print(" - Green dots: Human annotation.")
print(" - Pink dots: Model annotation")
print(" - White dots: Human and model annotate the same pixel.")

join_list = [utils.join_masks(pred_train_list[i], msk_list[i]) for i in range(len(msk_list))]
utils.plot_image_and_mask(img_names = file_names, img_list = img_list,
                        msk_names = jac_str, msk_list = join_list,
                        msk_head = 'Jaccard Similarity')  


utils.saveMasks(pred_train_list, file_ids, filetype = 'tif')
    


sss