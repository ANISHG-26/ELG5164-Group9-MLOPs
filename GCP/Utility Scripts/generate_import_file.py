from google.cloud import storage
from google.oauth2 import service_account
import os

# Set the path to your service account key file
key_path = 'beaming-team-376517-82a98807d500.json'

# Set the environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

# Authenticate using the key file
credentials = service_account.Credentials.from_service_account_file(key_path)

# Define your GCP Cloud Storage bucket and folder paths
bucket_name = 'cloud-ai-platform-54bd831b-8d3c-431d-acbc-4f155b670016'

# Create a GCP Cloud Storage client
storage_client = storage.Client(credentials=credentials)

# Get a reference to the GCP Cloud Storage bucket
bucket = storage_client.bucket(bucket_name)
print("Bucket Obtained")

# List the objects in the bucket
blobs = bucket.list_blobs()
data = {}
for blob in blobs:
    label = blob.name.split('/')[1]
    data["gs://"+bucket_name+"/"+blob.name] = label

with open('garbage_dataset_importfile.csv', 'w') as f:
    for key in data.keys():
        f.write("%s,%s\n"%(key,data[key]))