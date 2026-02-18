Available Apptainer images are:

* tramway-hpc-250124-py3{[8](https://dl.pasteur.fr/fop/fnQY9aI4/tramway-hpc-250124-py38.sif),[9](https://dl.pasteur.fr/fop/bqIvcQOu/tramway-hpc-250124-py39.sif),[10](https://dl.pasteur.fr/fop/CGSBWArs/tramway-hpc-250124-py310.sif),[11](https://dl.pasteur.fr/fop/AFE8uuyA/tramway-hpc-250124-py311.sif),[12](https://dl.pasteur.fr/fop/4KuTx4gX/tramway-hpc-250124-py312.sif),[13](https://dl.pasteur.fr/fop/5XftLW2h/tramway-hpc-250124-py313.sif)}.sif

  *tramway0.6.9* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *8*, *9*, *10*, *11*, *12*, *13* depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.x -s -m tramway $@`.
  The included Python dependencies are the following:

```
certifi==2019.11.28
chardet==3.0.4
contourpy==1.1.1 # py == 3.8
contourpy==1.3.0 # py == 3.9
contourpy==1.3.1 # py >= 3.10
cvxopt==1.3.2
cycler==0.12.1
dbus-python==1.2.16
fonttools==4.55.5
h5py==3.11.0 # py == 3.8
h5py==3.12.1 # py >= 3.9
idna==2.8
imageio==2.35.1 # py == 3.8
imageio==2.37.0 # py >= 3.9
importlib_resources==6.4.5 # py == 3.8
importlib_resources==6.5.2 # py == 3.9
joblib==1.4.2
kiwisolver==1.4.7 # py <= 3.9
kiwisolver==1.4.8 # py >= 3.10
lazy_loader==0.4
matplotlib==3.7.5 # py == 3.8
matplotlib==3.9.4 # py == 3.9
matplotlib==3.10.0 # py >= 3.10
networkx==3.1 # py == 3.8
networkx==3.2.1 # py == 3.9
networkx==3.4.2 # py >= 3.10
numpy==1.24.4 # py == 3.8
numpy==2.0.2 # py == 3.9
numpy==2.2.2 # py >= 3.10
opencv-python==4.11.0.86
packaging==24.2
pandas==2.0.3 # py == 3.8
pandas==2.2.3 # py >= 3.9
pillow==10.4.0 # py == 3.8
pillow==11.1.0 # py >= 3.9
polytope==0.2.5
PyGObject==3.36.0
pyparsing==3.1.4 # py == 3.8
pyparsing==3.2.1 # py >= 3.9
python-apt==2.0.1+ubuntu0.20.4.1
python-dateutil==2.9.0.post0
pytz==2024.2
requests==2.22.0
requests-unixsocket==0.2.0
rwa-python==0.9.5
scikit-image==0.21.0 # py == 3.8
scikit-image==0.24.0 # py == 3.9
scikit-image==0.25.0 # py >= 3.10
scikit-learn==1.3.2 # py == 3.8
scikit-learn==1.6.1 # py >= 3.9
scipy==1.10.1 # py == 3.8
scipy==1.13.1 # py == 3.9
scipy==1.15.1 # py >= 3.10
setuptools==75.8.0 # py >= 3.11
six==1.17.0
stopit==1.1.2
threadpoolctl==3.5.0
tifffile==2023.7.10 # py == 3.8
tifffile==2024.8.30 # py == 3.9
tifffile==2025.1.10 # py >= 3.10
tqdm==4.67.1
tzdata==2025.1
urllib3==1.25.8
zipp==3.20.2 # py == 3.8
zipp==3.21.0 # py == 3.9
```

* tramway-hpc-260217-py3{[9](https://dl.pasteur.fr/fop/vk2BlcFV/tramway-hpc-260217-py39.sif),[10](https://dl.pasteur.fr/fop/vt7BJqDk/tramway-hpc-260217-py310.sif),[11](https://dl.pasteur.fr/fop/yZNQ1Wn4/tramway-hpc-260217-py311.sif),[12](https://dl.pasteur.fr/fop/X4Pqd1Bk/tramway-hpc-260217-py312.sif),[13](https://dl.pasteur.fr/fop/FBEYEP0E/tramway-hpc-260217-py313.sif),[14](https://dl.pasteur.fr/fop/NbJmUfmQ/tramway-hpc-260217-py314.sif)}.sif

  *tramway0.7* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *9*, *10*, *11*, *12*, *13*, *14* depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.x -s -m tramway $@`.
  The included Python dependencies are the following:

```
blinker==1.4  # py >= 3.10
certifi==2019.11.28  # py == 3.9
chardet==3.0.4  # py == 3.9
contourpy==1.3.0  # py == 3.9
contourpy==1.3.2  # py == 3.10
contourpy==1.3.3  # py >= 3.11
cryptography==3.4.8  # py >= 3.10
cvxopt==1.3.3
cycler==0.12.1
dbus-python==1.2.16  # py == 3.9
dbus-python==1.2.18  # py >= 3.10
dill==0.4.1
distro==1.7.0  # py >= 3.10
exceptiongroup==1.3.1  # py == 3.9
fonttools==4.60.2  # py == 3.9
fonttools==4.61.1  # py >= 3.10
h5py==3.14.0  # py == 3.9
h5py==3.15.1  # py >= 3.10
httplib2==0.20.2  # py >=3.10
idna==2.8  # py == 3.9
ImageIO==2.37.2
importlib_resources==6.5.2  # py == 3.9
importlib-metadata==4.6.4  # py >= 3.10
iniconfig==2.1.0  # py == 3.9
jeepney==0.7.1  # py >= 3.10
joblib==1.5.3
keyring==23.5.0  # py >= 3.10
kiwisolver==1.4.7  # py == 3.9
kiwisolver==1.4.9  # py >= 3.10
launchpadlib==1.10.16  # py >= 3.10
lazr.restfulclient==0.14.4  # py >= 3.10
lazr.uri==1.0.6  # py >= 3.10
lazy_loader==0.4
matplotlib==3.9.4  # py == 3.9
matplotlib==3.10.8  # py >= 3.10
more-itertools==8.10.0  # py >= 3.10
multiprocess==0.70.19
networkx==3.2.1  # py == 3.9
networkx==3.4.2  # py == 3.10
networkx==3.6.1  # py >= 3.11
numpy==2.0.2  # py == 3.9
numpy==2.2.6  # py == 3.10
numpy==2.4.2  # py >= 3.11
oauthlib==3.2.0  # py >= 3.10
opencv-python==4.13.0.92
packaging==26.0
pandas==2.3.3
pillow==11.3.0  # py == 3.9
pillow==12.1.1  # py >= 3.10
pluggy==1.6.0  # py == 3.9
polytope==0.2.5
Pygments==2.19.2  # py == 3.9
PyGObject==3.36.0  # py == 3.9
PyGObject==3.42.1  # py >= 3.10
PyJWT==2.3.0  # py >= 3.10
pyparsing==3.3.2
pytest==8.4.2  # py == 3.9
python-apt==2.0.1+ubuntu0.20.4.1  # py == 3.9
python-apt==2.4.0+ubuntu4.1  # py >= 3.10
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.22.0  # py == 3.9
requests-unixsocket==0.2.0  # py == 3.9
rwa-python==0.10
scikit-image==0.24.0  # py == 3.9
scikit-image==0.25.2  # py == 3.10
scikit-image==0.26.0  # py >= 3.11
scikit-learn==1.6.1  # py == 3.9
scikit-learn==1.7.2  # py == 3.10
scikit-learn==1.8.0  # py >= 3.11
scipy==1.13.1  # py == 3.9
scipy==1.15.3  # py == 3.10
scipy==1.17.0  # py >= 3.11
SecretStorage==3.3.1  # py >= 3.10
setuptools==82.0.0  # py >= 3.12
six==1.17.0  # py == 3.9
six==1.16.0  # py >= 3.10
stopit==1.1.2
threadpoolctl==3.6.0
tifffile==2024.8.30  # py == 3.9
tifffile==2025.5.10  # py == 3.10
tifffile==2026.2.16  # py >= 3.11
tomli==2.4.0  # py == 3.9
tqdm==4.67.3
typing_extensions==4.15.0  # py == 3.9
tzdata==2025.3
urllib3==1.25.8  # py == 3.9
wadllib==1.3.6  # py >= 3.10
zipp==3.23.0  # py == 3.9
zipp==1.0.0  # py >= 3.10
```
