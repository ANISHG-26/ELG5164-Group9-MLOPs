import os
import boto3
from decouple import config


# Set the name of the bucket and the path to the folder to upload
bucket_name = 'garbagenet-bucket-29032023'

os.environ['AWS_ACCESS_KEY_ID'] = config('ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = config('SECRET_KEY')

# Create an S3 client
s3 = boto3.client('s3')

subfolders = ['cardboard','glass','metal','paper', 'plastic','trash']
# Loop through all the files in your local folder structure and its subfolders
dirname = os.path.dirname(__file__)
current_directory = os.getcwd()
for subfolder in subfolders:
    # Define the path of the subdirectory
    subfolder_path = os.path.join('.\Garbage Dataset', subfolder)
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            # Get the local file path
            local_file_path = os.path.join(root, file)
            s3_path = os.path.join('Garbage Dataset/', '{}/'.format(subfolder), file)
            # Upload the file to S3
            s3.upload_file(local_file_path, bucket_name, s3_path)
            print(f'{local_file_path} has been uploaded to {bucket_name}.')