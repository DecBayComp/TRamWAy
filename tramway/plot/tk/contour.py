# -*- coding: utf-8 -*-

# Copyright © 2017, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import matplotlib
matplotlib.use('TkAgg')

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd
from tramway.io import *
from tramway.helper.tesselation import find_imt
from ..contour import ContourEditor
from tkinter import BooleanVar, IntVar, DoubleVar, StringVar
import tkinter as tk
import tkinter.filedialog as filedialog
from tkinter import ttk
import os.path
import sys
import traceback



class TkContourEditor(FigureCanvasTkAgg):
	def __init__(self, parent, **kwargs):
		FigureCanvasTkAgg.__init__(self, Figure(), master=parent, **kwargs)
		self.plt = ContourEditor(self.figure, renderer=self)
		self.cell = IntVar()
		self.step = IntVar()
		self.variable = StringVar()
		self.integral = DoubleVar()
		self.cell.trace('w', self._cell)
		self.step.trace('w', self._step)
		self.variable.trace('w', self._variable)
		self.figure.canvas.mpl_connect('button_press_event', self.onclick)
		def _pass(*args, **kwargs):
			pass
		self.cell._report_exception = _pass
		self.step._report_exception = _pass
		self.plt.delaunay = True
		self.plt.callback = self.integral.set
		self.label = None

	def _cell(self, *args):
		c = self.cell.get()
		try:
			self.plt.cell = c
		except ValueError:
			self.cell.set(-1)
			raise
		except:
			print(traceback.format_exc())
			raise

	def _step(self, *args):
		s = self.step.get()
		if s < 1:
			self.step.set(1)
		else:
			try:
				self.plt.step = s
			except:
				print(traceback.format_exc())
				raise

	def _variable(self, *args):
		try:
			self.plt.variable = self.variable.get()
			if self.label is not None:
				if self.plt._variables[self.plt.variable][1:]:
					self.label['text'] = 'Curl:'
				else:
					self.label['text'] = 'Mean:'
		except:
			print(traceback.format_exc())
			raise

	def onclick(self, event):
		x, y = event.xdata, event.ydata
		if x is None or y is None:
			return
		self.cell.set(self.plt.find_cell((x, y)))

	def toggle_delaunay(self, *args):
		self.plt.delaunay = not self.plt._delaunay

	@property
	def cells(self):
		raise RuntimeError("'cells' attribute is read-only")

	@cells.setter
	def cells(self, cs):
		try:
			self.plt.cells = cs
		except:
			print(traceback.format_exc())
			raise

	@property
	def map(self):
		raise RuntimeError("'map' attribute is read-only")

	@map.setter
	def map(self, m):
		try:
			self.plt.map = m
		except:
			print(traceback.format_exc())
			raise

	@property
	def variables(self):
		return self.plt.variables

	@property
	def debug(self):
		return self.plt.debug

	@debug.setter
	def debug(self, d):
		self.plt.debug = d



class FileChooser(tk.Frame):
	def __init__(self, parent, filetypes={}, **kwargs):
		tk.Frame.__init__(self, parent, **kwargs)
		self.filetypes = filetypes
		self.filepath = StringVar()
		self.rowconfigure(0, weight=1)
		self.columnconfigure(0, weight=1)
		self.columnconfigure(1, weight=1)
		self.label = tk.Label(self, textvariable=self.filepath)
		self.label.grid(row=0, column=0, sticky=tk.W+tk.E)
		self.button = tk.Button(self, text='Open', command=self.open_input_file)
		self.button.grid(row=0, column=1, sticky=tk.E)

	def open_input_file(self):
		filepath = filedialog.askopenfilename(filetypes=self.filetypes)
		if filepath:
			self.filepath.set(filepath)



class ContourEditingApp(tk.Frame):
	def __init__(self, parent=None):
		tk.Frame.__init__(self, parent)
		self.create_widgets()
		self.input_file.filepath.trace('w', self.new_map)
		self.editor.step.set(1)
		self.editor.label = self.integral_label

	def create_widgets(self):
		self.input_file = StringVar()

		#self.rowconfigure(0, weight=1)
		self.columnconfigure(0, weight=1)
		self.rowconfigure(1, weight=1)
		self.columnconfigure(1, weight=1)
		self.input_file = FileChooser(self, (('RWA files', '*.rwa'), ('All files', '*.*')),
			relief=tk.RIDGE, borderwidth=2) # test border
		self.input_file.grid(row=0, column=0, columnspan=2, sticky=tk.W+tk.E+tk.N)
		self.editor = TkContourEditor(self)
		self.editor.get_tk_widget().grid(row=1, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
		self.side_panel = tk.Frame(self, relief=tk.RIDGE, borderwidth=2)
		self.side_panel.grid(row=1, column=1, sticky=tk.E+tk.N+tk.S, padx=2, pady=2)

		r = c = 1
		self.cell_label = tk.Label(self.side_panel, text='Cell:')
		self.cell_label.grid(column=c, row=r, sticky=tk.W)
		self.cell_input = tk.Entry(self.side_panel, width=4, textvariable=self.editor.cell)
		self.cell_input.grid(column=c+1, row=r, sticky=tk.W)
		r += 1
		self.step_label = tk.Label(self.side_panel, text='Step:')
		self.step_label.grid(column=c, row=r, sticky=tk.W)
		self.step_input = tk.Spinbox(self.side_panel, width=2, from_=1, to=99, increment=1,
			textvariable=self.editor.step)
		self.step_input.grid(column=c+1, row=r, sticky=tk.W)
		r += 1
		self.variable_label = tk.Label(self.side_panel, text='Variable:')
		self.variable_label.grid(column=c, row=r, sticky=tk.W)
		self.variable_input = ttk.Combobox(self.side_panel, state='readonly',
			textvariable=self.editor.variable)
		self.variable_input.grid(column=c+1, row=r, sticky=tk.W)
		r += 1
		self.integral_label = tk.Label(self.side_panel, text='Sum:')
		self.integral_label.grid(column=c, row=r, sticky=tk.W)
		self.integral_output = tk.Label(self.side_panel, textvariable=self.editor.integral)
		self.integral_output.grid(column=c+1, row=r, sticky=tk.W)
		r += 1
		self.delaunay_button = tk.Button(self.side_panel, text='Toggle Delaunay',
			command=self.editor.toggle_delaunay)
		self.delaunay_button.grid(column=c, columnspan=2, row=r, sticky=tk.W)
		#for _r in range(r):
		#	self.side_panel.rowconfigure(_r, weight=1)
		for _c in range(c):
			self.side_panel.columnconfigure(_c, weight=1)
		for child in self.side_panel.winfo_children():
			child.grid_configure(padx=2, pady=2)
		self.disable_widgets()


	def new_map(self, *args, **kwargs):
		map_file = self.input_file.filepath.get()
		store = HDF5Store(map_file, 'r')
		try:
			mode = store.peek('mode')
			if mode == '(callable)':
				maps = store.peek('result')
			else:
				maps = store.peek(mode)
		except KeyError:
			raise ValueError('not a map file: {}'.format(map_file))
			return
		else:
			try:
				tess_file = store.peek('rwa_file')
			except KeyError:
				# old format
				tess_file = store.peek('imt_file')
		finally:
			store.close()
		if not isinstance(tess_file, str):
			tess_file = tess_file.decode('utf-8')
		tess_file = os.path.join(os.path.dirname(map_file), tess_file)
		self.disable_widgets()
		store = HDF5Store(tess_file, 'r')
		try:
			self.editor.cells = store.peek('cells')
		finally:
			store.close()
		self.editor.map = maps
		self.variable_input['values'] = self.editor.variables
		if not self.editor.variables[1:]:
			self.editor.variable.set(self.editor.variables[0])
		self.enable_widgets()

	def disable_widgets(self):
		self.cell_label['state'] = 'disabled'
		self.cell_input['state'] = 'disabled'
		self.variable_label['state'] = 'disabled'
		self.variable_input.state(('disabled',))
		self.editor.cell.set(-1)
		self.editor.map = None

	def enable_widgets(self):
		self.cell_label['state'] = 'normal'
		self.cell_input['state'] = 'normal'
		self.variable_label['state'] = 'normal'
		self.variable_input.state(('!disabled',))

	@property
	def debug(self):
		return self.editor.debug

	@debug.setter
	def debug(self, d):
		self.editor.debug = d



if __name__ == '__main__':
	root = tk.Tk()
	root.protocol('WM_DELETE_WINDOW', sys.exit)
	root.rowconfigure(0, weight=1)
	root.columnconfigure(0, weight=1)
	app = ContourEditingApp(root)
	app.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
	app.debug = '-d' in sys.argv or '--debug' in sys.argv
	app.mainloop()

