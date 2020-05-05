FROM tensorflow/tensorflow:1.4.1-gpu-py3

RUN pip3 install --upgrade pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

#RUN sudo apt-get install nvidia-docker2
#RUN sudo pkill -SIGHUP dockerd

#VOLUME /notebook
#WORKDIR /notebook
#EXPOSE 8888
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root 