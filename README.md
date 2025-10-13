INSTALLATION 

1 - Create a dedicated conda environment to avoid package conflicts:

conda create --name wlcl_env


2 - Go in the environment, install newest version of pymc:

conda activate wlcl_env

conda install -c conda-forge pymc


3 - Install wlclusters (pyproject-based):

git clone https://github.com/LorisChappuis/wlclusters.git

cd wlclusters

pip install .

Alternatively, build a wheel/sdist locally and install:

pip install build

python -m build

pip install dist/*.whl


4 - Adding the environment as a jupyter notebook kernel :

conda install -c anaconda ipykernel

python -m ipykernel install --user --name=wlcl_env



5 - install ipywidgets for better pymc visualisation in jupyter-notebook (optional)

conda install ipywidgets
