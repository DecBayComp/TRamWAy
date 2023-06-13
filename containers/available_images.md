Available singularity images are:

* [tramway-hpc-210720.sif](http://dl.pasteur.fr/fop/uHLVsQi0/tramway-hpc-210720.sif)

  *tramway0.6* image with target *roi* and *animate* similar to previous images.
  Python is available as *python3.6* and the run command admits option *-s*, passed to Python,
  resulting in command `python3.6 -s -m tramway $@`.
  The included Python dependencies are the following:

```
bcrypt==3.2.0
cached-property==1.5.2
cffi==1.14.6
cryptography==3.4.7
cvxopt==1.2.6
cycler==0.10.0
decorator==4.4.2
h5py==3.1.0
imageio==2.9.0
joblib==1.0.1
kiwisolver==1.3.1
matplotlib==3.3.4
networkx==2.5.1
numpy==1.19.5
opencv-python==4.5.3.56
pandas==1.1.5
paramiko==2.7.2
Pillow==8.3.1
polytope==0.2.3
pycparser==2.20
pycurl==7.43.0
pygobject==3.20.0
PyNaCl==1.4.0
pyparsing==2.4.7
python-apt==1.1.0b1+ubuntu0.16.4.12
python-dateutil==2.8.2
pytz==2021.1
PyWavelets==1.1.1
rwa-python==0.8.5
scikit-image==0.17.2
scikit-learn==0.24.2
scipy==1.5.4
six==1.16.0
stopit==1.1.2
threadpoolctl==2.2.0
tifffile==2020.9.3
tqdm==4.61.2
```


* [tramway-hpc-210815-py36.sif](http://dl.pasteur.fr/fop/ak21MN70/tramway-hpc-210815-py36.sif)

  *tramway0.6.2* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.6* and the run command admits option *-s*, passed to Python,
  resulting in command `python3.6 -s -m tramway $@`.
  The included Python dependencies are the following:

```
cached-property==1.5.2
cvxopt==1.2.6
cycler==0.10.0
decorator==4.4.2
h5py==3.1.0
imageio==2.9.0
joblib==1.0.1
kiwisolver==1.3.1
matplotlib==3.3.4
networkx==2.5.1
numpy==1.19.5
opencv-python==4.5.3.56
pandas==1.1.5
Pillow==8.3.1
polytope==0.2.3
pycurl==7.43.0
PyGObject==3.20.0
pyparsing==2.4.7
python-apt==1.1.0b1+ubuntu0.16.4.12
python-dateutil==2.8.2
pytz==2021.1
PyWavelets==1.1.1
rwa-python==0.8.5
scikit-image==0.17.2
scikit-learn==0.24.2
scipy==1.5.4
six==1.16.0
stopit==1.1.2
threadpoolctl==2.2.0
tifffile==2020.9.3
tqdm==4.62.1
```

* tramway-hpc-210815-py3{[7](http://dl.pasteur.fr/fop/SWFP6UYv/tramway-hpc-210815-py37.sif),[8](http://dl.pasteur.fr/fop/isOFAOAQ/tramway-hpc-210815-py38.sif),[9](http://dl.pasteur.fr/fop/zG31eu1j/tramway-hpc-210815-py39.sif)}.sif

  *tramway0.6.2* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *7*, *8*, *9*, depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.8 -s -m tramway $@`.
  The included Python dependencies are the following:

```
cached-property==1.5.2  # py37 only
cvxopt==1.2.6
cycler==0.10.0
h5py==3.3.0
imageio==2.9.0
joblib==1.0.1
kiwisolver==1.3.1
matplotlib==3.4.3
networkx==2.6.2
numpy==1.21.2
opencv-python==4.5.3.56
pandas==1.3.2
Pillow==8.3.1
polytope==0.2.3
pycurl==7.43.0
PyGObject==3.20.0
pyparsing==2.4.7
python-apt==1.1.0b1+ubuntu0.16.4.12
python-dateutil==2.8.2
pytz==2021.1
PyWavelets==1.1.1
rwa-python==0.8.5
scikit-image==0.18.2
scikit-learn==0.24.2
scipy==1.7.1
six==1.16.0
stopit==1.1.2
threadpoolctl==2.2.0
tifffile==2021.8.8
tqdm==4.62.1
```


* [tramway-hpc-210902-py36.sif](http://dl.pasteur.fr/fop/XLrSl3jX/tramway-hpc-210902-py36.sif)

  *tramway0.6.3* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.6* and the run command admits option *-s*, passed to Python,
  resulting in command `python3.6 -s -m tramway $@`.
  The included Python dependencies are the following:

```
cached-property==1.5.2
cvxopt==1.2.6
cycler==0.10.0
decorator==4.4.2
h5py==3.1.0
imageio==2.9.0
joblib==1.0.1
kiwisolver==1.3.1
matplotlib==3.3.4
networkx==2.5.1
numpy==1.19.5
opencv-python==4.5.3.56
pandas==1.1.5
Pillow==8.3.2
polytope==0.2.3
pycurl==7.43.0
PyGObject==3.20.0
pyparsing==2.4.7
python-apt==1.1.0b1+ubuntu0.16.4.12
python-dateutil==2.8.2
pytz==2021.1
PyWavelets==1.1.1
rwa-python==0.8.5
scikit-image==0.17.2
scikit-learn==0.24.2
scipy==1.5.4
six==1.16.0
stopit==1.1.2
threadpoolctl==2.2.0
tifffile==2020.9.3
tqdm==4.62.2
```


* tramway-hpc-210902-py3{[7](http://dl.pasteur.fr/fop/xWrWWsEA/tramway-hpc-210902-py37.sif),[8](http://dl.pasteur.fr/fop/VTRrmm3o/tramway-hpc-210902-py38.sif),[9](http://dl.pasteur.fr/fop/xywwERaD/tramway-hpc-210902-py39.sif)}.sif

  *tramway0.6.3* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *7*, *8*, *9*, depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.8 -s -m tramway $@`.
  The included Python dependencies are the following:

```
cached-property==1.5.2  # py37 only
cvxopt==1.2.6
cycler==0.10.0
h5py==3.4.0
imageio==2.9.0
joblib==1.0.1
kiwisolver==1.3.2
matplotlib==3.4.3
networkx==2.6.2
numpy==1.21.2
opencv-python==4.5.3.56
pandas==1.3.2
Pillow==8.3.2
polytope==0.2.3
pycurl==7.43.0
PyGObject==3.20.0
pyparsing==2.4.7
python-apt==1.1.0b1+ubuntu0.16.4.12
python-dateutil==2.8.2
pytz==2021.1
PyWavelets==1.1.1
rwa-python==0.8.5
scikit-image==0.18.3
scikit-learn==0.24.2
scipy==1.7.1
six==1.16.0
stopit==1.1.2
threadpoolctl==2.2.0
tifffile==2021.8.30
tqdm==4.62.2
```


* tramway-hpc-220312-py3{[6](http://dl.pasteur.fr/fop/ePQViHBR/tramway-hpc-220312-py36.sif),[7](http://dl.pasteur.fr/fop/tTm5DW5L/tramway-hpc-220312-py37.sif),[8](http://dl.pasteur.fr/fop/G5nyISvN/tramway-hpc-220312-py38.sif),[9](http://dl.pasteur.fr/fop/8gfNt53O/tramway-hpc-220312-py39.sif),[10](http://dl.pasteur.fr/fop/8CIf4PIS/tramway-hpc-220312-py310.sif)}.sif

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

* tramway-hpc-230609-py3{[8](https://dl.pasteur.fr/fop/74lfEWxO/tramway-hpc-230609-py38.sif),[9](https://dl.pasteur.fr/fop/XgQ06STg/tramway-hpc-230609-py39.sif),[10](https://dl.pasteur.fr/fop/cjIlKNPy/tramway-hpc-230609-py310.sif),[11](https://dl.pasteur.fr/fop/fIpRkzfG/tramway-hpc-230609-py311.sif)}.sif

  *tramway0.6.7* image with targets *hpc-minimal* and *animate*, and *scikit-learn* as an additional package.
  Python is available as *python3.x*, with *x* any of *8*, *9*, *10*, *11* depending on the container file.
  The run command admits option *-s*, passed to Python,
  resulting in command (e.g.) `python3.8 -s -m tramway $@`.
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
