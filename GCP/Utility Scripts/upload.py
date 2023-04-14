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
gcp_folder_path = 'Garbage classification/'

# Create a GCP Cloud Storage client
storage_client = storage.Client(credentials=credentials)

# Get a list of buckets
''''
buckets = storage_client.list_buckets()
for bucket in buckets:
    print(bucket.name)
'''

# Get a reference to the GCP Cloud Storage bucket
bucket = storage_client.bucket(bucket_name)
print("Bucket Obtained")

# List the objects in the bucket
'''
blobs = bucket.list_blobs()
for blob in blobs:
    print(blob.name)
'''

subfolders = ['cardboard','glass','metal','paper', 'plastic','trash']
# Loop through all the files in your local folder structure and its subfolders
dirname = os.path.dirname(__file__)
current_directory = os.getcwd()
for subfolder in subfolders:
    # Define the path of the subdirectory
    subfolder_path = os.path.join('.\Garbage classification', subfolder)
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            # Get the local file path
            local_file_path = os.path.join(root, file)
            
            #print("Local File Path ",local_file_path)
           
            # Get the corresponding GCP Cloud Storage file path
            gcs_file_path = os.path.join(gcp_folder_path, '{}/'.format(subfolder), file)

            
            # Check if the file exists in GCP Cloud Storage
            blob = bucket.blob(gcs_file_path)
            if not blob.exists():
                # If the file does not exist, upload it to GCP Cloud Storage
                os.chdir(subfolder_path)
                blob.upload_from_filename(file)
                os.chdir(current_directory)
                print(f'{local_file_path} uploaded to {gcs_file_path} in GCP Cloud Storage')
            else:
                print(f'{gcs_file_path} already exists in GCP Cloud Storage')