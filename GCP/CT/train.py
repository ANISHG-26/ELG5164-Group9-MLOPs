# Imports
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd 
import os
import zipfile
import datetime
from io import BytesIO
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from keras.preprocessing import image as keras_image_preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Constants for GCP
BUCKET_NAME = "cloud-ai-platform-54bd831b-8d3c-431d-acbc-4f155b670016"
GCS_DATA_FILE_PATH = "gs://cloud-ai-platform-54bd831b-8d3c-431d-acbc-4f155b670016/FilePath/garbage_dataset_importfile.csv"
PROJECT_ID = "beaming-team-376517"
KEY_PATH = 'beaming-team-376517-82a98807d500.json'

# Constants for ML
IMAGE_WIDTH = 224    
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_SHAPE = (224,224,3)
NUM_CLASSES = 6
categories = {'paper': 0,'cardboard': 1,'plastic': 2,'metal': 3,'trash': 4,'glass': 5}
BATCH_SIZE = 16
EPOCHS = 15
MODEL_DIR = "models/"

# Custom utilities
def zipfolder(foldername, filename, includeEmptyDir=True):   
    empty_dirs = []  
    zip = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED)  
    for root, dirs, files in os.walk(foldername):  
        empty_dirs.extend([dir for dir in dirs if os.listdir(os.path.join(root, dir)) == []])  
        for name in files:  
            zip.write(os.path.join(root ,name))  
        if includeEmptyDir:  
            for dir in empty_dirs:  
                zif = zipfile.ZipInfo(os.path.join(root, dir) + "/")  
                zip.writestr(zif, "")  
        empty_dirs = []  
    zip.close()
    
# Custom utilities for model training
def custom_gcp_dataframe_iterator(df, batch_size, bucket):
    num_classes = len(df['label'].unique())
    while True:
        # iterate over batches of the dataframe
        for i in range(0, len(df), batch_size):
            # get the batch of file paths and labels
            batch_df = df.iloc[i:i+batch_size]
            batch_paths = batch_df['image_gcp_location'].values
            batch_labels = batch_df['label'].values
            # load and preprocess the images in the batch
            batch_images = []
            for gcs_path in batch_paths:
                path = "/".join(gcs_path.split('/')[3:])
                blob = bucket.blob(path)
                gcs_image_bytes = blob.download_as_bytes()
                gcs_image = Image.open(BytesIO(gcs_image_bytes)).convert('RGB')
                gcs_image = gcs_image.resize((224, 224))
                gcs_image = np.array(gcs_image).astype('float32') / 255.0
                batch_images.append(gcs_image)
            # Yield the preprocessed images and one-hot encoded labels as a batch
            yield np.array(batch_images), to_categorical(batch_labels, num_classes=num_classes)

# GCP Set Up
def get_gcp_bucket():
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    storage_client = storage.Client(project=PROJECT_ID,credentials=credentials)
    bucket = storage_client.bucket(BUCKET_NAME)
    return bucket



# Data preparation for custom training
def data_prep():
    gcs_file_path = "FilePath/garbage_dataset_importfile.csv"
    bucket = get_gcp_bucket()
    blob = bucket.blob(gcs_file_path)
    blob.download_to_filename('data.csv')
    df = pd.read_csv("data.csv",header=None)
    df.columns = ['image_gcp_location', 'label']
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Change the categories from numbers to names
    df["label"] = df["label"].map(categories).astype(str)
    
    # We first split the data into two sets and then split the validate_df to two sets
    train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
    validate_df, test_df = train_test_split(validate_df, test_size=0.3, random_state=42)

    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    total_train = train_df.shape[0]
    total_test = test_df.shape[0]
    total_validate = validate_df.shape[0]
    
    print("#################### DATA METRICS ####################")
    print('train size = ', total_train, 'validate size = ', total_validate, 'test size = ', total_test)
    
    return (train_df,validate_df,total_train,total_validate)

# Building Tensorflow Model
def get_model(image_shape,num_classes):
    
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
            
def train():
    print("#################### TRAINING STARTS ####################")
    
    # Getting the data
    train_df, validate_df, total_train, total_validate = data_prep()
    
    # Getting the model
    model = get_model(IMAGE_SHAPE,NUM_CLASSES)
    
    # Create model save directory
    model_save_path = os.path.join(os.getcwd(), MODEL_DIR)
    os.mkdir(model_save_path) 
    print("Directory '% s' created" % MODEL_DIR)

    bucket = get_gcp_bucket() 

    train_generator = custom_gcp_dataframe_iterator(train_df,BATCH_SIZE,bucket)
    validation_generator = custom_gcp_dataframe_iterator(validate_df,BATCH_SIZE,bucket)
    
    # Model Training
    history = model.fit_generator(
                generator=train_generator, 
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps=total_validate//BATCH_SIZE,
                steps_per_epoch=total_train//BATCH_SIZE,
                #callbacks=callbacks
            )
    
    CONCRETE_INPUT = "numpy_inputs"
    
    # Tensorflow serving utilities
    def _preprocess(bytes_input):
        decoded = tf.io.decode_jpeg(bytes_input, channels=3)
        decoded = tf.image.convert_image_dtype(decoded, tf.float32)
        resized = tf.image.resize(decoded, size=(224, 224))
        return resized


    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def preprocess_fn(bytes_inputs):
        decoded_images = tf.map_fn(
            _preprocess, bytes_inputs, dtype=tf.float32, back_prop=False
        )
        return {
            CONCRETE_INPUT: decoded_images
        }  # User needs to make sure the key matches model's input


    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def serving_fn(bytes_inputs):
        images = preprocess_fn(bytes_inputs)
        prob = m_call(**images)
        return prob


    m_call = tf.function(model.call).get_concrete_function(
        [tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32, name=CONCRETE_INPUT)]
    )

    tf.saved_model.save(model, model_save_path, signatures={"serving_default": serving_fn})
    
    ct = datetime.datetime.now()
    ct_string = ct.strftime('%Y-%m-%d %H:%M:%S')
    
    zipfile_name = 'models'+ct_string+'.zip'
    zipfolder(MODEL_DIR,zipfile_name)
    gcs_upload_path = "Trained_models/" + zipfile_name
    upload_blob('models.zip', gcs_upload_path)
    
    
    
def upload_blob(source_file_name, destination_blob_name):
    bucket = get_gcp_bucket() 
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
    
if __name__ == '__main__':
    print('main')
    train()
