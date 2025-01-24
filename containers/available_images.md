Available singularity images are:

* tramway-hpc-220312-py3{6,7,[8](http://dl.pasteur.fr/fop/n2SSFwr5/tramway-hpc-220312-py38.sif),[9](http://dl.pasteur.fr/fop/W6GcBTqk/tramway-hpc-220312-py39.sif),[10](http://dl.pasteur.fr/fop/Cmmobhil/tramway-hpc-220312-py310.sif)}.sif

  *tramway0.6.4* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *6*, *7*, *8*, *9*, *10* depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.8 -s -m tramway $@`.
  The included Python dependencies are the following:

```
cached-property==1.5.2 # <=py38
certifi==2019.11.28
chardet==3.0.4
cvxopt==1.3.0
cycler==0.11.0
dbus-python==1.2.16
decorator==4.4.2 # py36 only
fonttools==4.30.0 # >=py37
h5py==3.1.0 # py36 only
h5py==3.6.0 # >=py37
idna==2.8
imageio==2.15.0 # py36 only
imageio==2.16.1 # >=py37
importlib-resources==5.4.0 # py36 only
joblib==1.1.0
kiwisolver==1.3.1 # py36 only
kiwisolver==1.3.2 # >=py37
matplotlib==3.3.4 # py36 only
matplotlib==3.5.1 # >=py37
networkx==2.5.1 # py36 only
networkx==2.6.3 # py37 only
networkx==2.7.1 # >=py38
numpy==1.19.5 # py36 only
numpy==1.21.5 # py37 only
numpy==1.22.3 # >=py38
opencv-python==4.5.5.64
packaging==21.3 # >=py37
pandas==1.1.5 # py36 only
pandas==1.3.5 # py37 only
pandas==1.4.1 # >=py38
Pillow==8.4.0 # py36 only
Pillow==9.0.1 # >=py37
polytope==0.2.3
PyGObject==3.36.0
pyparsing==3.0.7
python-apt==2.0.0+ubuntu0.20.4.7
python-dateutil==2.8.2
pytz==2021.3
PyWavelets==1.1.1 # py36 only
PyWavelets==1.3.0 # >=py37
requests==2.22.0
requests-unixsocket==0.2.0
rwa-python==0.9.1
scikit-image==0.17.2 # py36 only
scikit-image==0.19.2 # >=py37
scikit-learn==0.24.2 # py36 only
scikit-learn==1.0.2 # >=py37
scipy==1.5.4 # py36 only
scipy==1.7.3 # py37 only
scipy==1.8.0 # >=py38
six==1.14.0
stopit==1.1.2
threadpoolctl==3.1.0
tifffile==2020.9.3 # py36 only
tifffile==2021.11.2 # py37 only
tifffile==2022.2.9 # >=py38
tqdm==4.63.0
urllib3==1.25.8
zipp==3.6.0 # py36 only
```

* tramway-hpc-230609-py3{8,9,[10](https://dl.pasteur.fr/fop/keDoKINW/tramway-hpc-230609-py310.sif),[11](https://dl.pasteur.fr/fop/znbYr11f/tramway-hpc-230609-py311.sif)}.sif

  *tramway0.6.7* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *8*, *9*, *10*, *11* depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.x -s -m tramway $@`.
  The included Python dependencies are the following:

```
certifi==2019.11.28
chardet==3.0.4
contourpy==1.0.7
cvxopt==1.3.1
cycler==0.11.0
dbus-python==1.2.16
fonttools==4.40.0
h5py==3.8.0
idna==2.8
imageio==2.31.1
importlib-resources==5.12.0 # py<310
joblib==1.2.0
kiwisolver==1.4.4
lazy_loader==0.2
matplotlib==3.7.1
networkx==3.1
numpy==1.24.3
opencv-python==4.7.0.72
packaging==23.1
pandas==2.0.2
Pillow==9.5.0
polytope==0.2.3
PyGObject==3.36.0
pyparsing==3.0.9
python-apt==2.0.1+ubuntu0.20.4.1
python-dateutil==2.8.2
pytz==2023.3
PyWavelets==1.4.1
requests==2.22.0
requests-unixsocket==0.2.0
rwa-python==0.9.3
scikit-image==0.20.0
scikit-learn==1.2.2
scipy==1.9.1 # py<310
scipy==1.10.1 # py>39
six==1.16.0
stopit==1.1.2
threadpoolctl==3.1.0
tifffile==2023.4.12
tqdm==4.65.0
tzdata==2023.3
urllib3==1.25.8
zipp==3.15.0 # py<310
```

* tramway-hpc-240423-py3{[8](https://dl.pasteur.fr/fop/B3LCHpTL/tramway-hpc-240423-py38.sif),[9](https://dl.pasteur.fr/fop/crbAgtuu/tramway-hpc-240423-py39.sif),[10](https://dl.pasteur.fr/fop/zZ2nrnKP/tramway-hpc-240423-py310.sif),[11](https://dl.pasteur.fr/fop/7F8PPD7j/tramway-hpc-240423-py311.sif),[12](https://dl.pasteur.fr/fop/lRWEB8oR/tramway-hpc-240423-py312.sif)}.sif

  *tramway0.6.8* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *8*, *9*, *10*, *11*, *12* depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.x -s -m tramway $@`.
  The included Python dependencies are the following:

```
certifi==2019.11.28
chardet==3.0.4
contourpy==1.1.1 # py38 only
contourpy==1.2.1 # py>38
cvxopt==1.3.2
cycler==0.12.1
dbus-python==1.2.16
fonttools==4.51.0
h5py==3.11.0
idna==2.8
imageio==2.34.1
importlib-resources==6.4.0 # py<312
joblib==1.4.0
kiwisolver==1.4.5
lazy_loader==0.4
matplotlib==3.7.5 # py38 only
matplotlib==3.8.4 # py>38
networkx==3.1 # py38 only
networkx==3.2.1 # py39 only
networkx==3.3 # py>39
numpy==1.24.4 # py38 only
numpy==1.26.4 # py>38
opencv-python==4.9.0.80
packaging==24.0
pandas==2.0.3 # py38 only
pandas==2.2.2 # py>38
pillow==10.3.0
polytope==0.2.5
PyGObject==3.36.0
pyparsing==3.1.2
python-apt==2.0.1+ubuntu0.20.4.1
python-dateutil==2.9.0.post0
pytz==2024.1
PyWavelets==1.4.1
requests==2.22.0
requests-unixsocket==0.2.0
rwa-python==0.9.3
scikit-image==0.21.0 # py38 only
scikit-image==0.22.0 # py39 only
scikit-image==0.23.2 # py>39
scikit-learn==1.3.2 # py38 only
scikit-learn==1.4.2 # py>38
scipy==1.10.1 # py38 only
scipy==1.13.0 # py>38
setuptools==69.5.1 # py312 only
six==1.16.0
stopit==1.1.2
threadpoolctl==3.4.0
tifffile==2023.7.10 # py38 only
tifffile==2024.4.18 # py>38
tqdm==4.66.2
tzdata==2024.1
urllib3==1.25.8
wheel==0.43.0 # py312 only
zipp==3.18.1 # py<310 only
```

* tramway-hpc-250124-py3{[8](https://dl.pasteur.fr/fop/xVjMGhTa/tramway-hpc-2501024-py38.sif),[9](https://dl.pasteur.fr/fop/AyK3JTC0/tramway-hpc-2501024-py39.sif),[10](https://dl.pasteur.fr/fop/jEmrwWhp/tramway-hpc-2501024-py310.sif),[11](https://dl.pasteur.fr/fop/QmL7TE6w/tramway-hpc-2501024-py311.sif),[12](https://dl.pasteur.fr/fop/elytCrRB/tramway-hpc-2501024-py312.sif),[13](https://dl.pasteur.fr/fop/hSeIMslv/tramway-hpc-2501024-py313.sif)}.sif

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

