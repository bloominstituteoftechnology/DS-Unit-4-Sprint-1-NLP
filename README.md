## NLP-Setup environment with Anaconda

1. Clone the repo to your local: `git clone https://github.com/skhabiri/ML-NLP.git`
For package management we are going to use conda.
2. Create the virtual environment: cd ML_NLP; `conda create -n ML_NLP python==3.7`
Now the environment is created in ~/opt/anaconda3/envs/ML_NLP. We can see the list of conda environments with `conda env list`. To remove an environment `conda env remove --name <conda-env>`. 
3. For installing the required packages, first we need to activate the environment `conda activate ML_NLP`. 
4. Install the packages listed in requirements.txt `pip install -r requirements.txt`. To list the installed packages in the environment: `conda list`. 


√ ML_NLP % cat requirements.txt
gensim==3.8.1
pyLDAvis==2.1.2
spacy==2.2.3
scikit-learn==0.22.2
seaborn==0.9.0
squarify==0.4.3
ipykernel
nltk
pandas
scipy
beautifulsoup4

5. ipykernel is Ipython Kernel, a python execution backend for Jupyter. In order to open the python environment from jupyter we add an ipython kernel referencing to conda environment by `python -m ipykernel install --user --name ML_NLP --display-name "ML_NLP (Python3.7)"`.
6. Next we need to download and install the models for spacy: `python -m spacy download en_core_web_md`, `python -m spacy download en_core_web_lg`
Now you can deactivate the environment and launch jupyter lab and select ML_NLP as the Ipython kernel.
