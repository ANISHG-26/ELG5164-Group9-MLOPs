import os
from azure.storage.blob import BlobServiceClient
from decouple import config


# Getting connection string and container name
container_name = config('CONTAINER_NAME')
connection_string = config('CONNECTION_STRING')

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create the container client
container_client = blob_service_client.get_container_client(container_name)

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
            
            # Create a BlobClient for the file we want to upload
            blob_client = container_client.get_blob_client(local_file_path)

            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            print(f'{local_file_path} has been uploaded to {container_name}.')