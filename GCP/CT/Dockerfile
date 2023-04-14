FROM tensorflow/tensorflow:2.11.0
WORKDIR /root

COPY requirements.txt ./requirements.txt
COPY train.py ./train.py
COPY beaming-team-376517-82a98807d500.json ./beaming-team-376517-82a98807d500.json

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "train.py"]
