FROM tensorflow/tensorflow:2.11.0
WORKDIR /root

ENV CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=garbagenetclas4048036826;AccountKey=EqBkL/lJUkc01dqCxs9FOrQ1qlqZBdjLYMg2kuhhs1Cx945LjUpDyAtReCNkVUThBeyF6Qnmw6re+AStO0YfxA==;EndpointSuffix=core.windows.net"
ENV CONTAINER_NAME="azureml-blobstore-afaa340e-8029-4073-bfbd-01de34e364bd"

COPY requirements.txt ./requirements.txt
COPY train.py ./train.py

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "train.py"]
