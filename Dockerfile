FROM jupyter/minimal-notebook:latest

# Install Requirements
COPY requirements.txt /tmp/

RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    @jupyterlab/git \
    @jupyterlab/toc \
    @jupyterlab/github 
