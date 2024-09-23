#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import shutil
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
# Importing dependencies

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import *
import keras_cv


# In[2]:


current_working_directory = os.getcwd()

# print output to the console
print(current_working_directory)


# In[3]:


csv_path = 'C:/Users/Admin/project/dataset/csv/meta.csv'
df_meta = pd.read_csv(csv_path)
dicom_data = pd.read_csv('C:/Users/Admin/project/dataset/csv/dicom_info.csv')


# In[4]:


image_dir = 'C:/Users/Admin/project/dataset/jpeg'
full_mammogram_images = dicom_data[dicom_data.SeriesDescription == 'full mammogram images'].image_path
cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
roi_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

full_mammogram_images = full_mammogram_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
roi_mask_images = roi_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
full_mammogram_images.iloc[0]


# In[5]:


full_mammogram_dict = dict()
cropped_dict = dict()
roi_mask_dict = dict()

for dicom in full_mammogram_images:
    # print(dicom)
    key = dicom.split("/")[6]
    # print(key)
    full_mammogram_dict[key] = dicom
for dicom in cropped_images:
    key = dicom.split("/")[6]
    cropped_dict[key] = dicom
for dicom in roi_mask_images:
    key = dicom.split("/")[6]
    roi_mask_dict[key] = dicom


# In[6]:


mass_train_data = pd.read_csv('C:/Users/Admin/project/dataset/csv/mass_case_description_train_set.csv')
mass_test_data = pd.read_csv('C:/Users/Admin/project/dataset/csv/mass_case_description_test_set.csv')
calc_train_data = pd.read_csv('C:/Users/Admin/project/dataset/csv/calc_case_description_train_set.csv')
calc_test_data = pd.read_csv('C:/Users/Admin/project/dataset/csv/calc_case_description_test_set.csv')


# In[7]:


def filter_dataframe_by_base_directory(df):
    base_directory = '/content/jpeg'

    # Check if all three columns start with the base directory
    mask = (
        df['image file path'].str.startswith(base_directory) &
        df['cropped image file path'].str.startswith(base_directory) &
        df['ROI mask file path'].str.startswith(base_directory)
    )

    # Keep only the rows where all three columns start with the base directory
    filtered_df = df[mask]

    return filtered_df


# In[8]:


def fix_image_path_mass(dataset):
    for i, img in enumerate(dataset.values):
        img_name = img[11].split("/")[2]
        if img_name in full_mammogram_dict:
            dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        img_name = img[12].split("/")[2]
        if img_name in cropped_dict:
            dataset.iloc[i, 12] = cropped_dict[img_name]

        img_name = img[13].split("/")[2]
        if img_name in roi_mask_dict:
            dataset.iloc[i, 13] = roi_mask_dict[img_name]


# In[9]:


def fix_image_path_mass(dataset):
    for i, img in enumerate(dataset.values):
        img_name = img[11].split("/")[2]
        if img_name in full_mammogram_dict:
            dataset.iloc[i, 11] = full_mammogram_dict[img_name]

        img_name = img[12].split("/")[2]
        if img_name in cropped_dict:
            dataset.iloc[i, 12] = cropped_dict[img_name]

        img_name = img[13].split("/")[2]
        if img_name in roi_mask_dict:
            dataset.iloc[i, 13] = roi_mask_dict[img_name]


# In[10]:


fix_image_path_mass(mass_test_data)
fix_image_path_mass(mass_train_data)
mass_train = mass_train_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})
mass_test = mass_test_data.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})


# In[11]:


shape=mass_train.mass_shape
pd.value_counts(shape)
import pandas as pd
class_counts = mass_train['pathology'].value_counts()
least = class_counts[class_counts<81].index.tolist()
mass_train['pathology']=mass_train['pathology'].apply(lambda x:'others' if x in least else x)
print(mass_train['pathology'].value_counts())
mass_train['pathology'] = mass_train['pathology'].astype(str)


# In[14]:


import pandas as pd

# Convert any float values in the mass_shape column to strings
mass_train['pathology'] = mass_train['pathology'].astype(str)

# Group by the patient ID and aggregate the mass_shape values
combined_pathology = mass_train.groupby('patient_id')['pathology'].agg(','.join).reset_index()

# Merge the aggregated mass_shape values back to the original DataFrame based on patient ID

mass_train_combined = pd.merge(mass_train, combined_pathology, on='patient_id', suffixes=('', '_combined'))

# Drop the original mass_shape column
mass_train_combined.drop('pathology', axis=1, inplace=True)

# Rename the combined mass_shape column
mass_train_combined.rename(columns={'pathology_combined': 'pathology'}, inplace=True)


# In[15]:


train_data=mass_train_combined


# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[29]:


data=mass_train
data['pathology'] = data['pathology'].astype(str)


# In[30]:


shape1=mass_test.pathology
pd.value_counts(shape1)


# In[31]:



import pandas as pd
class_counts = mass_test['pathology'].value_counts()
least = class_counts[class_counts<40].index.tolist()
mass_test['pathology']=mass_test['pathology'].apply(lambda x:'others' if x in least else x)
print(mass_test['pathology'].value_counts())
mass_test['pathology'] = mass_test['pathology'].astype(str)


# In[32]:


import pandas as pd

# Convert any float values in the mass_shape column to strings
mass_test['pathology'] = mass_test['pathology'].astype(str)

# Group by the patient ID and aggregate the mass_shape values
combined_mass_shapes_test = mass_test.groupby('patient_id')['pathology'].agg(','.join).reset_index()

# Merge the aggregated mass_shape values back to the original DataFrame based on patient ID
mass_test_combined = pd.merge(mass_test, combined_mass_shapes_test, on='patient_id', suffixes=('', '_combined'))

# Drop the original mass_shape column
mass_test_combined.drop('pathology', axis=1, inplace=True)

# Rename the combined mass_shape column
mass_test_combined.rename(columns={'pathology_combined': 'pathology'}, inplace=True)


# In[33]:


test_data=mass_test_combined


# In[34]:


mass_test_combined


# In[35]:


mass_test_combined['image_file_path']


# In[37]:


# Create an ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
)

# Create training and validation data generators
train_generator = datagen.flow_from_dataframe(
    dataframe=data,
    x_col='image_file_path',
    y_col='pathology',
    target_size=(512, 512),
    color_mode='grayscale',  # Set color_mode to 'grayscale' for single-channel images
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
# Model Building
num_classes = mass_test_combined['pathology'].nunique()


# In[38]:


base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(512, 512, 1))  # Set input_shape to (512, 512, 1)

# Add custom classification layers on top of the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(mass_train['pathology'].unique()), activation='sigmoid')(x)  # Use sigmoid for multi-label classification

# Combine base model and custom layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# In[39]:


# Train the model with early stopping
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)


# In[40]:


history = model.fit(
    train_generator,
    epochs=20,  
    callbacks=[early_stopping]
)


# In[41]:


data2 =mass_test_combined


# In[42]:


data2['pathology']


# In[43]:


test_generator = datagen.flow_from_dataframe(
    dataframe=mass_test_combined,
    x_col='image_file_path',
    y_col='pathology',
    target_size=(512, 512),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)


# In[44]:


import matplotlib.pyplot as plt

# Extract unique class names from the mass_shape column
class_names = mass_train['pathology'].unique().tolist()


# In[45]:



class_names


# In[46]:


#Create a dictionary mapping numerical labels to class names
label_to_class = {i: class_names[i] for i in range(len(class_names))}

# Function to decode predicted classes
def decode_predictions(predictions, threshold=0.2):
    decoded_labels = []
    for pred in predictions:
        decoded_labels.append([label_to_class[i] for i, p in enumerate(pred) if p >= threshold])
    return decoded_labels

# Prepare test data generator (assuming you have already defined test_generator)
test_generator.reset()  # Reset the generator to the beginning
images, true_labels = next(test_generator)  # Get a batch of images and true labels


# In[47]:


true_labels


# In[48]:


# Take only 5 images and their true labels
images = images[:5]
true_labels = true_labels[:5]

# Make predictions on the batch of images
predictions = model.predict(images)

# Decode predicted classes
decoded_predictions = decode_predictions(predictions)

# Convert true labels from numerical to class names
true_labels = [label_to_class[label[0]] for label in true_labels]

# Display the images along with predicted and true labels in rows
num_images = len(images)
fig, axes = plt.subplots(num_images, 1, figsize=(5, 5 * num_images))

for i in range(num_images):
    axes[i].imshow(images[i])
    axes[i].set_title(f"Predicted: {decoded_predictions[i]}\nTrue: {true_labels[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# In[ ]:




