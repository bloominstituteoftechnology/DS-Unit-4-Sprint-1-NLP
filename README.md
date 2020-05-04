# DS-Unit-4-Sprint-1-NLP

Hello World!! 

Good Morning! I'm excited to have you in Unit 4. We'll be kicking-off with some NLP. This week we will be working locally with conda's environments.  Take a few minutes to fork the repository and follow the instructions to set up this week's environment.
Conda Environments
You will be completing each module this sprint on your machine. We will be using conda environments to manage the packages and their dependencies for this sprint's content. In a classroom setting, instructors typically abstract away environment for you. However, environment management is an important professional data science skill. We showed you how to manage environments using pipvirtual env during Unit 3, but in this sprint, we will introduce an environment management tool common in the data science community:
conda: Package, dependency and environment management for any language—Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN, and more.
The easiest way to install conda on your machine is via the Anaconda Distribution of Python & R. Once you have conda installed, read "A Guide to Conda Environments". This article will provide an introduce into some of the conda basics. If you need some additional help getting started, the official "Setting started with conda" guide will point you in the right direction.
:snake:
To get the sprint environment setup:
Open your command line tool (Terminal for MacOS, Anaconda Prompt for Windows)
Navigate to the folder with this sprint's content. There should be a requirements.txt
Run conda create -n U4-S1-NLP python==3.7 => You can also rename the environment if you would like. Once the command completes, your conda environment should be ready.
Now, we are going to add in the require python packages for this sprint. You will need to 'activate' the conda environment: source activate U4-S1-NLP on Terminal or conda activate U4-S1-NLP on Anaconda Prompt. Once your environment is activate, run pip install -r requirements.txt which will install the required packages into your environment.
We are going to also add an Ipython Kernel reference to your conda environment, so we can use it from JupyterLab.
Next run python -m ipykernel install --user --name U4-S1-NLP --display-name "U4-S1-NLP (Python3)" => This will add a json object to an ipython file, so JupterLab will know that it can use this isolated instance of Python. :)
Last step, we need to install the models for Spacy. Run these commands python -m spacy download en_core_web_md and python -m spacy download en_core_web_lg
Deactivate your conda environment and launch JupyterLab. You should know see "U4-S1-NLP (Python3)" in the list of available kernels on launch screen.