FROM jupyter/datascience-notebook:latest
# Alternate pinned source:
# FROM docker.io/paperspace/tensorflow:1.5.0-gpu

# Install Requirements
COPY requirements.txt /tmp/

RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN python3 -m spacy download en_core_web_md

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/git \
    @jupyterlab/toc \
    @jupyterlab/github 
