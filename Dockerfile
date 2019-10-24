FROM gcr.io/tensorflow/tensorflow:2.0.0-gpu
# Alternate pinned source:
# FROM docker.io/paperspace/tensorflow:1.5.0-gpu

RUN mv /usr/local/bin/pip /usr/local/bin/pip_2

RUN apt-get -y update && apt-get install -y python3-pip && pip3 install --upgrade pip

RUN rm /usr/local/bin/pip && mv /usr/local/bin/pip_2 /usr/local/bin/pip

# Install Requirements
COPY requirements.txt /tmp/

RUN pip3 install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN python3 -m spacy download en_core_web_md

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/git \
    @jupyterlab/toc \
    @jupyterlab/github 
