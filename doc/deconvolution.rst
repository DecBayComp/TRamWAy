.. _deconvolution:

The :mod:`~tramway.deconvolution` module adds several dependencies, including *scikit-image>=0.14.2*, *tifffile* and *tensorflow*. *keras* is optional as it is now included in *tensorflow* releases.

*tifffile* will in turn require *imagecodecs*, which builds smoothly only with the most recent version of many libraries. 

Installation notes about these dependencies are available on https://pypi.org/project/imagecodecs/ 
In addition, on "old" Ubuntu versions such as *18.04*, you may need to select past versions of *imagecodecs* and *tifffile*, e.g. (working example on *18.04.3 LTS*)::

    pip3 install --user imagecodecs==2019.2.22
    pip3 install --user tifffile==2019.3.18

The :mod:`~tramway.deconvolution` module can be called as a command in two ways::

    python3 -m tramway.deconvolution.inference -h
    python3 -m tramway.utils.deconvolve -h

Each command is independent and exhibits slightly different arguments.

For now, the multi-gpu feature is set to use 4 GPUs, which should be available otherwise the program will crash.
