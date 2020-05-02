FROM tensorflow/tensorflow:1.15.2

RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

#VOLUME /notebook
#WORKDIR /notebook
#EXPOSE 8888
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root 