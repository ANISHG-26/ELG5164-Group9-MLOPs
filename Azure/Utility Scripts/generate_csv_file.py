from azure.storage.blob import BlobServiceClient
from decouple import config

# Getting connection string and container name
container_name = config('CONTAINER_NAME')
connection_string = config('CONNECTION_STRING')

# Create the BlobServiceClient object which will be used to create a container client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create the container client
container_client = blob_service_client.get_container_client(container_name)

# List the objects in the bucket
blobs = container_client.list_blobs()
data = {}
for blob in blobs:
    label = blob.name.split('/')[1]
    data["https://"+container_name+"/"+blob.name] = label

with open('garbage_dataset_importfile.csv', 'w') as f:
    for key in data.keys():
        f.write("%s,%s\n"%(key,data[key]))
