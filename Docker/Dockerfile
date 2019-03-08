# The offical tensorflow docker package

FROM nvcr.io/nvidia/tensorflow:18.07-py3

# Install extra packages if required

RUN apt-get update && apt-get install -y libpq-dev build-essential





# Set the working directory inside the container to /app

WORKDIR /app


RUN pip install scipy Pillow scikit-image imgaug sklearn pandas
RUN pip install opencv-python opencv-contrib-python
RUN pip install lifelines


RUN apt-get install -y  libsm6 libxrender1 libfontconfig1
RUN apt-get update && apt-get install -y libsm6 libxext6



