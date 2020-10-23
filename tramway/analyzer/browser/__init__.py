
from ..attribute import *

class Browser(AnalyzerNode):
    """
    `~tramway.analyzer.RWAnalyzer.browser` attribute of an :class:`~tramway.analyzer.RWAnalyzer`.

    Rendering is based on *bokeh* and is performed in a web browser tab.
    An optional side panel with experimental export features is shown if
    argument ``side_panel=True`` is passed to `show_maps` or argument ``webdriver``
    is defined:

    .. code-block:: python

        from tramway.analyzer import *

        a = RWAnalyzer()

        a.spt_data.from_rwa_files('*.rwa')

        from selenium import webdriver

        a.browser.show_maps(webdriver=webdriver.Firefox)

    Defining a webdriver is required for exporting figures.
    Supported file formats are *.png* and *.svg*.
    Note that package `selenium <https://selenium-python.readthedocs.io/installation.html>`_ is required.

    The above example can be explicitly run with ``bokeh serve``,
    or else it can be run with the standard Python interpreter or in a Jupyter notebook
    as long as the *script* attribute is defined:

    .. code-block:: python

        from tramway.analyzer import *

        a = RWAnalyzer()

        a.spt_data.from_rwa_files('*.rwa')

        try:
            a.script = __file__
        except NameError: # in a notebook
            a.script = 'MyNotebook.ipynb' # this notebook's name; please adapt

        from selenium import webdriver

        a.browser.show_maps(webdriver=webdriver.Firefox)

    The later example will call ``bokeh serve --show`` on the file specified in the *script* attribute.

    The showed parameter values can also be exported with the side panel.
    Note that all features are exported together with the spatial bin center coordinates.
    """
    def show_maps(self, **kwargs):
        """ see also :func:`~tramway.plot.bokeh.analyzer.browse_maps`. """
        from tramway.plot.bokeh.analyzer import browse_maps
        browse_maps(self._eldest_parent, **kwargs)


__all__ = ['Browser']

