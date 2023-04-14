import boto3
from decouple import config
import os


# Set the name of the bucket and the path to the folder to upload
bucket_name = 'garbagenet-bucket-29032023'

os.environ['AWS_ACCESS_KEY_ID'] = config('ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = config('SECRET_KEY')

# Create an S3 client
s3 = boto3.client('s3')
data = {}

# List the objects in the bucket

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket_name)
for page in pages:
    for s3_object in page['Contents']:
        s3_file_path = s3_object['Key']
        label = s3_file_path.split('/')[1]
        data["s3://"+bucket_name+"/"+s3_file_path] = label

with open('garbage_dataset_importfile.csv', 'w') as f:
    for key in data.keys():
        f.write("%s,%s\n"%(key,data[key]))