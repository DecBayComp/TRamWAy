
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
    __slots__ = ('_colormap','_clim','_this_notebook_cell_only')
    @property
    def colormap(self):
        """ *str*: Colormap for inferred parameter maps.

        See also :func:`~tramway.plot.bokeh.map.scalar_map_2d`.
        """
        return self._colormap
    @colormap.setter
    def colormap(self, cm):
        self._colormap = cm
    @property
    def clim(self):
        """
        *dict* of 2-element *array-like*:
            Color lower and upper values (dict values) for each feature (dict keys)
        """
        return self._clim
    @clim.setter
    def clim(self, clim):
        self._clim = clim
    def __init__(self, analyzer):
        AnalyzerNode.__init__(self, parent=analyzer)
        self._colormap = None
        self._clim = None
        self._this_notebook_cell_only = None
    @property
    def script(self):
        return self._parent.script
    @script.setter
    def script(self, path):
        self._parent.script = path
    @property
    def this_notebook_cell_only(self):
        return self._this_notebook_cell_only
    @this_notebook_cell_only.setter
    def this_notebook_cell_only(self, cell_index):
        self._this_notebook_cell_only = cell_index
    def show_maps(self, **kwargs):
        """ See also :func:`~tramway.plot.bokeh.analyzer.browse_maps`. """
        if self.this_notebook_cell_only and \
                self.script is not None and \
                self.script.endswith('.ipynb'):
            self.logger.info("""\
the designated .ipynb file will be exported;
be sure to save the notebook after any modification
so that the changes are reported in the .ipynb file
""")
            # export the .ipynb file to a Python script
            script_path = self.script
            script_content = import_ipynb(script_path)
            # discard the other code cells
            script_content = split_cells(script_content)
            if isinstance(self.this_notebook_cell_only, bool):
                caller_cells = []
                for _, cell in script_content:
                    for line in cell:
                        line = line.split('#')[0]
                        if '.browser.show_maps(' in line:
                            # a cell may appear several times
                            caller_cells.append(cell)
                if not caller_cells:
                    raise RuntimeError("cannot find an expression calling `browser.show_maps`")
                if caller_cells[1:]:
                    raise RuntimeError('multiple `browser.show_maps` calls found; please specify the index of the notebook cell')
                caller_cell = caller_cells[0]
            else:
                notebook_cell_index = self.this_notebook_cell_only
                try:
                    _, caller_cell = script_content[notebook_cell_index]
                except IndexError:
                    for i, cell in enumerate(script_content):
                        print(f'notebook cell {i}:\n')
                        print(''.join(cell[1]))
                    raise
                else:
                    self.logger.debug(f'notebook cell index {notebook_cell_index}:\n\n'+''.join(caller_cell))
            condensed_code = []
            for line in caller_cell:
                line = line.split('#')[0].strip()
                if line and \
                        'script' not in line and \
                        'this_notebook_cell_only' not in line:
                    condensed_code.append(line+'\n')
            # run bokeh serve on the exported notebook cell
            import tempfile, os, subprocess, sys
            f, tmpfile = tempfile.mkstemp(suffix='.py')
            try:
                for line in condensed_code:
                    os.write(f, line.encode('utf-8'))
                os.close(f)
                self.logger.info(f'running: python -m bokeh serve --show {tmpfile}')
                p = subprocess.Popen([sys.executable, '-m',
                        'bokeh', 'serve', '--show', tmpfile],
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    close_fds=True,
                    encoding='utf-8')
                self.logger.info('interrupt the kernel to stop bokeh and proceed back in the notebook')
                try:
                    out, err = p.communicate()
                except KeyboardInterrupt:
                    out, err = p.communicate()
                if err:
                    self.logger.error(err)
                else:
                    if out:
                        self.logger.debug(out)
            finally:
                try:
                    os.unlink(tmpfile)
                except FileNotFoundError:
                    pass
        else:
            from tramway.plot.bokeh.analyzer import browse_maps
            browse_maps(self._eldest_parent, **kwargs)

Attribute.register(Browser)


def import_ipynb(path):
    from tramway.analyzer.env.environments import Env
    return Env().import_ipynb(path)

def split_cells(script):
    import re
    cells = []
    cell_index = None
    cell = []
    match = True
    for line in script:
        line_ = line.rstrip()
        match_with_index = re.fullmatch(r'# In\[(?P<cell_index>[1-9][0-9]*)\]:', line_)
        match_without_index = line_ == r'# In[ ]:'
        match = match_with_index or match_without_index
        if match:
            if cells or cell_index is not None or cell:
                cells.append((cell_index, cell))
            if match_with_index:
                cell_index = int(match_with_index.group('cell_index'))
            cell = []
        cell.append(line)
    if not match:
        cells.append((cell_index, cell))
    return cells


__all__ = ['Browser']

