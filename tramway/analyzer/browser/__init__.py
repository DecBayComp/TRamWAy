
from ..attribute import *

class Browser(AnalyzerNode):
    """
    :attr:`~tramway.analyzer.RWAnalyzer.browser` attribute of an
    :class:`~tramway.analyzer.RWAnalyzer` object.

    Rendering is based on *bokeh* and is performed in a web browser tab.
    An optional side panel with experimental export features is shown if
    argument ``side_panel=True`` is passed to :meth:`show_maps` or argument
    `webdriver` is defined:

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
    __slots__ = ('_colormap',)
    @property
    def colormap(self):
        """ *str*: Colormap for inferred parameter maps.

        See also :func:`~tramway.plot.bokeh.map.scalar_map_2d`."""
        return self._colormap
    @colormap.setter
    def colormap(self, cm):
        self._colormap = cm
    def __init__(self, analyzer):
        AnalyzerNode.__init__(self, parent=analyzer)
        self._colormap = None
    def show_maps(self, **kwargs):
        """ See also :func:`~tramway.plot.bokeh.analyzer.browse_maps`. """
        from tramway.plot.bokeh.analyzer import browse_maps
        browse_maps(self._eldest_parent, **kwargs)

Attribute.register(Browser)


__all__ = ['Browser']

