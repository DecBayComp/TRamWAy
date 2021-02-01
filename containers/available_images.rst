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

