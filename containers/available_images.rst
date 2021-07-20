Available singularity images are:

* `tramway-hpc-200928.sif <http://dl.pasteur.fr/fop/VsJYgkxP/tramway-hpc-200928.sif>`_

  minimal TRamWAy installation with target *roi* plus *tqdm*, *paramiko* and *scikit-learn*.
  Python is available as *python3.6* and the run command admits option *-s* to be passed to Python,
  resulting in command ``python3.6 -s -m tramway $@``.
  Available until 2021-03-12.

* `tramway-hpc-210125.sif <http://dl.pasteur.fr/fop/6Avu9HuV/tramway-hpc-210125.sif>`_

  similar to tramway-hpc-200928.sif with *tramway0.5* and additional dependency *stopit*.
  Available until 2021-03-12.

* `tramway-hpc-210201.sif <http://dl.pasteur.fr/fop/MSRwa8CR/tramway-hpc-210201.sif>`_

  *tramway0.5.1* image with target *roi* and *animate*, plus a few optional dependencies
  including *paramiko*, *scikit-learn* and *stopit*.
  Python is available as *python3.6* and the run command admits option *-s*, passed to Python,
  resulting in command ``python3.6 -s -m tramway $@``.

* `tramway-hpc-210222.sif <http://dl.pasteur.fr/fop/rzx2LnjB/tramway-hpc-210222.sif>`_

  pre-*tramway0.5.2* image with target *roi* and *animate*, plus a few optional dependencies
  including *paramiko*, *scikit-learn* and *stopit*.
  Python is available as *python3.6* and the run command admits option *-s*, passed to Python,
  resulting in command ``python3.6 -s -m tramway $@``.

* `tramway-hpc-210302.sif <http://dl.pasteur.fr/fop/53bfSkmM/tramway-hpc-210302.sif>`_

  *tramway0.5.2* image with target *roi* and *animate*, plus a few optional dependencies
  including *paramiko*, *scikit-learn* and *stopit*.
  Python is available as *python3.6* and the run command admits option *-s*, passed to Python,
  resulting in command ``python3.6 -s -m tramway $@``.

* `tramway-hpc-210720.sif <>`_

  *tramway0.6* image similar to previous images.
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
