FROM tensorflow/tensorflow:2.11.0
WORKDIR /root

ENV ACCESS_KEY="AKIA5F3ECBCQQ2SWFMG7"
ENV SECRET_KEY="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"

COPY requirements.txt ./requirements.txt
COPY train.py ./train.py

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "train.py"]
