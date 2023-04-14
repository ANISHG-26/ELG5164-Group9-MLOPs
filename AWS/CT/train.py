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
import boto3
from sklearn.model_selection import train_test_split
from keras.preprocessing import image as keras_image_preprocessing
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

# Constants for AWS
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('SECRET_KEY')
BUCKET_NAME = "garbagenet-bucket-30032023"
S3_DATA_FILE_PATH = 'FilePath/garbage_dataset_importfile.csv'

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
def custom_aws_dataframe_iterator(df, batch_size, s3_client):
    num_classes = len(df['label'].unique())
    while True:
        # iterate over batches of the dataframe
        for i in range(0, len(df), batch_size):
            # get the batch of file paths and labels
            batch_df = df.iloc[i:i+batch_size]
            batch_paths = batch_df['image_aws_location'].values
            batch_labels = batch_df['label'].values
            # load and preprocess the images in the batch
            batch_images = []
            for s3_object_path in batch_paths:
                s3_key = s3_object_path.split('/', 3)[3]
                response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
                s3_image = Image.open(BytesIO(response['Body'].read())).convert('RGB')
                s3_image = s3_image.resize((224, 224))
                s3_image = np.array(s3_image).astype('float32') / 255.0
                batch_images.append(s3_image)
            # Yield the preprocessed images and one-hot encoded labels as a batch
            yield np.array(batch_images), to_categorical(batch_labels, num_classes=num_classes)

# AWS Set Up
def get_aws_s3_client():
  s3_client = boto3.client('s3')
  return s3_client

# Data preparation for custom training
def data_prep():
    s3_client = get_aws_s3_client()
    
    # Set the local file path to save the downloaded object in the current directory
    local_data_file_path = os.path.join(os.getcwd(), 'data.csv')

    # Download the object from the bucket to a local file
    s3_client.download_file(BUCKET_NAME, S3_DATA_FILE_PATH, local_data_file_path)
    
    df = pd.read_csv("data.csv",header=None)
    df.columns = ['image_aws_location', 'label']
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

    s3_client = get_aws_s3_client()

    train_generator = custom_aws_dataframe_iterator(train_df,BATCH_SIZE,s3_client)
    validation_generator = custom_aws_dataframe_iterator(validate_df,BATCH_SIZE,s3_client)
    
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

    # Create a new directory
    model_local_directory = "Trained_models"
    os.mkdir(model_local_directory)

    # Move a file into the new directory
    file_to_move = zipfile_name
    os.rename(file_to_move, os.path.join(model_local_directory, file_to_move))
    
    aws_s3_upload_path = "Trained_models/" + zipfile_name
    upload_blob(aws_s3_upload_path)  
    
def upload_blob(upload_path):
    s3_client = get_aws_s3_client()
    s3_client.upload_file(upload_path, BUCKET_NAME, upload_path)

    print(f'{upload_path} has been uploaded to {BUCKET_NAME}.')
    
if __name__ == '__main__':
    print('main')
    train()
