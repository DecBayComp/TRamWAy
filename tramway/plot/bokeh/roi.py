
from bokeh.models import ColumnDataSource, Slider, CustomJS, BoxZoomTool, Toggle
from bokeh.models.glyphs import Patch, Patches
from bokeh.models.ranges import Range1d
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from tramway.tessellation.base import Partition
from .map import *
import numpy as np
import itertools
import warnings
import polytope as pt


class RoiController(object):
    def __init__(self, roi_collection=None,
                patches_x=None, patches_y=None,
                xlim=None, ylim=None,
                properties=('line_color','line_width'),
                default_values=('black',1),
                active_values=('green',2),
                **patches_kwargs):
        self.active_roi = None
        self.dynamic_properties = properties
        self.property_default_values = default_values
        self.property_active_values = active_values
        if patches_x is None or patches_y is None:
            patches_x, patches_y = [], []
            for bounding_box in roi_collection.bounding_box():
                patch_x = bounding_box[[0,0,2,2]]
                patch_y = bounding_box[[1,3,3,1]]
                patches_x.append(patch_x)
                patches_y.append(patch_y)
        self._patches_kwargs = patches_kwargs
        patch_data = dict(xs=patches_x, ys=patches_y)
        patch_data.update({prop: [value for _ in patches_x]
                for prop, value in zip(self.dynamic_properties, self.property_default_values)})
        patch_data['active'] = [0 for _ in patches_x]
        self.patches = ColumnDataSource(patch_data)
        self.line = ColumnDataSource(dict(x=[0,0,np.nan,0,0,np.nan,0,0,np.nan,0,0], y=[0,0,np.nan,0,0,np.nan,0,0,np.nan,0,0]))
        self._xlim = xlim
        self._ylim = ylim
    @property
    def patches(self):
        return self._patches
    @patches.setter
    def patches(self, ps):
        self._patches = ps
        self._make_prop_setter()
    @property
    def independent_patches(self):
        return isinstance(self.patches, (list, tuple))
    def _make_prop_setter(self):
        if self.independent_patches:
            if not self.patches:
                raise ValueError('empty sequence of patches')
            sample_patch = patch[0]
            if isinstance(sample_patch, ColumnDataSource):
                def _set(patch, prop, val):
                    assert patch != 'all'
                    patch.data[prop] = val
            elif isinstance(sample_patch, Patch):
                def _set(patch, prop, val):
                    assert patch != 'all'
                    setattr(patch.glyph, prop, val)
            else:
                raise TypeError('wrong patch type')
        else:
            if isinstance(self.patches, ColumnDataSource):
                def _all_patches_get_prop(prop):
                    return self.patches.data[prop]
                def _all_patches_set_prop(prop, values):
                    self.patches.data[prop] = values
            elif isinstance(self.patches, Patches):
                def _all_patches_get_prop(prop):
                    return getattr(self.patches, prop)
                def _all_patches_set_prop(prop, values):
                    setattr(self.patches, prop, values)
            else:
                raise TypeError('wrong multi-patch type')
            def _set(patch_index, prop, val):
                _all_patches_prop = _all_patches_get_prop(prop)
                if patch_index == 'all':
                    _all_patches_set_prop(prop, [val for _ in _all_patches_prop])
                else:
                    _all_patches_prop[patch_index] = val
        self._set = _set
    def _set_props(self, patch, property_values):
        if self.independent_patches:
            if patch == 'all':
                for patch in self.patches:
                    for prop, val in zip(self.dynamic_properties, property_values):
                        self._set(patch, prop, val)
                return # stop here
            else:
                patch = self.patches[patch]
        for prop, val in zip(self.dynamic_properties, property_values):
            self._set(patch, prop, val)
    def set_active(self, new_active_roi):
        if new_active_roi != self.active_roi and self.active_roi is not None:
            self.unset_active(self.active_roi)
        self._set_props(new_active_roi, self.property_active_values)
        #
        x0, x3 = self.xlim
        y0, y3 = self.ylim
        xs, ys = self.patches.data['xs'][new_active_roi], self.patches.data['ys'][new_active_roi]
        x1, x2 = xs[0], xs[2]
        y1, y2 = ys[0], ys[1]
        xc, yc = .5*(x1+x2), .5*(y1+y2)
        self.line.data['x'] = [x0, x1, np.nan, x2, x3, np.nan, xc, xc, np.nan, xc, xc]
        self.line.data['y'] = [yc, yc, np.nan, yc, yc, np.nan, y0, y1, np.nan, y2, y3]
        #
        self.active_roi = new_active_roi
    def unset_active(self, roi):
        self._set_props(roi, self.property_default_values)
        if roi == self.active_roi or roi == 'all':
            self.active_roi = None
    def js_callback(self, roi_plot):
        default_props = dict(zip(self.dynamic_properties, self.property_default_values))
        active_props = dict(zip(self.dynamic_properties, self.property_active_values))
        return CustomJS(args=dict(patches=self.patches, lines=self.line, roi_plot=roi_plot), code="""
                var active = patches.data['active'][0];
                var line_color = patches.data['line_color'];
                var line_width = patches.data['line_width'];
                line_color[active] = '{}';
                line_width[active] = {};
                var new_active = this.value-1;
                line_color[new_active] = '{}';
                line_width[new_active] = {};
                patches.data['active'][0] = new_active;
                var x = patches.data['xs'][new_active];
                var xmin = x[0], xmax = x[2];
                var xc = 0.5 * (xmin + xmax);
                var y = patches.data['ys'][new_active];
                var ymin = y[0], ymax = y[1];
                var yc = 0.5 * (ymin + ymax);
                x = lines.data['x'];
                x[1] = xmin, x[3] = xmax;
                for (var i of [6,7,9,10]) {{
                    x[i] = xc;
                }}
                y = lines.data['y'];
                for (i of [0,1,3,4]) {{
                    y[i] = yc;
                }}
                y[7] = ymin, y[9] = ymax;
                var width = xmax-xmin, height = ymax-ymin;
                if (width < height) {{
                    xmin = xc - 0.5 * height;
                    xmax = xc + 0.5 * height;
                }} else if (height < width) {{
                    ymin = yc - 0.5 * width;
                    ymax = yc + 0.5 * width;
                }}
                roi_plot.x_range.start = xmin;
                roi_plot.x_range.end   = xmax;
                roi_plot.y_range.start = ymin;
                roi_plot.y_range.end   = ymax;
                lines.change.emit();
                patches.change.emit();
                roi_plot.change.emit();
            """.format(
                default_props['line_color'],
                default_props['line_width'],
                active_props['line_color'],
                active_props['line_width']))
    @property
    def patches_kwargs(self):
        return dict(xs='xs', ys='ys', line_color='line_color', line_width='line_width',
                    source=self.patches, **self._patches_kwargs)
    @property
    def line_kwargs(self):
        active_props = dict(zip(self.dynamic_properties, self.property_active_values))
        return dict(x='x', y='y', source=self.line, **active_props)
    @property
    def xlim(self):
        if self._xlim is None:
            xs = list(itertools.chain(*self.patches.data['xs']))
            self._xlim = min(xs), max(xs)
        return self._xlim
    @xlim.setter
    def xlim(self, xl):
        self._xlim = xl
    @property
    def ylim(self):
        if self._ylim is None:
            ys = list(itertools.chain(*self.patches.data['ys']))
            self._ylim = min(ys), max(ys)
        return self._ylim
    @ylim.setter
    def ylim(self, yl):
        self._ylim = yl

def match_aspect(xlim, ylim):
    width, height = xlim[1]-xlim[0], ylim[1]-ylim[0]
    if width < height:
        xc = .5*(xlim[0]+xlim[1])
        xlim = [ xc-.5*height, xc+.5*height ]
    elif height < width:
        yc = .5*(ylim[0]+ylim[1])
        ylim = [ yc-.5*width, yc+.5*width ]
    return xlim, ylim


class RoiCollection(object):
    def __init__(self, global_tessellation=None, connected_components=None,
            roi_tessellations=None, roi_meta_tessellation=None,
            bounding_boxes=None):
        self.global_tessellation = global_tessellation
        self.roi_tessellations = roi_tessellations
        self.roi_meta_tessellation = roi_meta_tessellation
        self.connected_components = connected_components
        self._bounding_boxes = bounding_boxes
    def bounding_box(self, margin=None):
        if self._bounding_boxes is None:
            bb = []
            if self.roi_meta_tessellation is not None:
                meta_mesh = self.roi_meta_tessellation
                for vertex_indices in meta_mesh.cell_vertices:
                    lower_bound = meta_mesh.vertices[vertex_indices[0]]
                    upper_bound = meta_mesh.vertices[vertex_indices[2]]
                    bounding_box = np.r_[lower_bound, upper_bound]
                    bb.append(bounding_box)
            elif self.roi_tessellations is not None:
                for roi in self.roi_tessellations:
                    lower_bound = roi.bounding_box[['x','y']]['min']
                    upper_bound = roi.bounding_box[['x','y']]['max']
                    bounding_box = np.r_[lower_bound, upper_bound]
                    bb.append(bounding_box)
            else:
                tessellation = self.global_tessellation
                for component in self.connected_components:
                    ## center
                    #centers = tessellation.cell_centers[component]
                    #center = np.mean(centers, axis=0)
                    # bounding box
                    vertex_indices = set()
                    for cell in component:
                        vertex_indices |= set(tessellation.cell_vertices[cell])
                    try:
                        vertex_indices.remove(-1)
                    except KeyError:
                        pass
                    vertices = tessellation.vertices[list(vertex_indices)]
                    lower_bound, upper_bound = np.min(vertices, axis=0), np.max(vertices, axis=0)
                    bounding_box = np.r_[lower_bound, upper_bound]
                    #
                    #bb.append((center, bounding_box))
                    bb.append(bounding_box)
            ## sanity check
            #polygons = [ pt.box2poly([[b[0],b[2]],[b[1],b[3]]]) for b in bb ]
            #if not all([ all([ pt.is_empty(p.intersect(q)) for q in polygons[i+1:] ]) for i, p in enumerate(polygons) ]):
            #    warnings.warn('some rois overlap', RuntimeWarning)
            #
            self._bounding_boxes = bb
        if margin:
            bb = []
            for bounding_box in self._bounding_boxes:
                lower_bound = bounding_box[[0,1]]
                upper_bound = bounding_box[[2,3]]
                bounding_box = np.r_[lower_bound - margin, upper_bound + margin]
                bb.append(bounding_box)
        else:
            bb = self._bounding_boxes
        return bb
    def __len__(self):
        if self._bounding_boxes is None:
            return len(self.connected_components)
        else:
            return len(self._bounding_boxes)


class RoiBrowser(object):
    def __init__(self, global_map_cells=None, global_map_values=None, # these two args first for bw comp
            roi_tessellations=None, roi_map_values=None, roi_meta_tessellation=None,
            **kwargs):
        self.global_map_cells  = global_map_cells
        self.global_map_values = global_map_values
        self.points = None if roi_meta_tessellation is None else roi_meta_tessellation.points
        tessellation = lambda a: None if a is None else a.tessellation
        self.roi_model = RoiCollection(
                global_tessellation=tessellation(global_map_cells),
                roi_tessellations=roi_tessellations,
                roi_meta_tessellation=tessellation(roi_meta_tessellation),
                **kwargs)
        self.roi_map_values = roi_map_values
        self.first_active_roi = 0

        # options for view elements
        self.full_fov_figure_kwargs = dict(toolbar_location='above', toolbar_sticky=False, match_aspect=True, tools='pan, wheel_zoom, reset')
        self.zooming_in_figure_kwargs = dict(toolbar_location=None, active_drag=None, match_aspect=True, tools='pan, wheel_zoom, reset')
        self.scalar_map_2d_kwargs = {}
        self.points_kwargs = dict(color='r', alpha=.1)
        self.trajectories_kwargs = dict(color='r', line_width=.5, line_alpha=.5, loc_alpha=.1, loc_size=6)
        self.slider_kwargs = dict(title='roi')

        self.roi_controller = RoiController(self.roi_model, fill_color=None)
        x, y = self.points['x'].values, self.points['y'].values
        self.roi_controller.xlim = [x.min(), x.max()]
        self.roi_controller.ylim = [y.min(), y.max()]

    @property
    def values(self): # for bw compatibility during code transition
        warnings.warn('deprecated attribute: values', RuntimeWarning)
        return self.global_map_cells

    @property
    def points(self):
        if self._points is None:
            if self.global_map_cells is not None:
                self._points = self.global_map_cells.points
        return self._points

    @points.setter
    def points(self, pts):
        self._points = pts

    @property
    def cells(self):
        warnings.warn('deprecated attribute: cells', RuntimeWarning)
        return self.global_map_cells

    @property
    def roi_tessellations(self):
        return self.roi_model.roi_tessellations

    def full_fov(self):
        ctrl = self.roi_controller
        full_fov_fig = figure(**self.full_fov_figure_kwargs)
        full_fov_fig.add_tools(BoxZoomTool(match_aspect=True))
        if self.global_map_values is not None:
            scalar_map_2d(self.global_map_cells, self.global_map_values,
                    figure=full_fov_fig, **self.scalar_map_2d_kwargs)
        elif False:
            _min, _max = zip(*[ (values.min(), values.max()) for values in self.roi_map_values ])
            clim = (min(_min), max(_max))
            for cells,values in zip(self.roi_tessellations,self.roi_map_values):
                scalar_map_2d(cells, values, figure=full_fov_fig, clim=clim,
                        **self.scalar_map_2d_kwargs)
        plot_points(self.points, figure=full_fov_fig, **self.points_kwargs)
        full_fov_fig.patches(**ctrl.patches_kwargs)
        full_fov_fig.line(**ctrl.line_kwargs)
        ctrl.unset_active('all')
        ctrl.set_active(self.first_active_roi)
        self.roi_view_full_fov = full_fov_fig
        return full_fov_fig

    def zooming_in(self):
        ctrl = self.roi_controller
        roi_bb = self.roi_model.bounding_box(margin=.1)[self.first_active_roi]
        xlim = roi_bb[[0,2]].tolist()
        ylim = roi_bb[[1,3]].tolist()
        xlim, ylim = match_aspect(xlim, ylim)
        zooming_in_fig = figure(**self.zooming_in_figure_kwargs)
        zooming_in_fig.add_tools(BoxZoomTool(match_aspect=True))
        if self.global_map_values is not None:
            scalar_map_2d(self.global_map_cells, self.global_map_values, figure=zooming_in_fig,
                    **self.scalar_map_2d_kwargs)
        else:
            for cells,values in zip(self.roi_tessellations,self.roi_map_values):
                scalar_map_2d(cells, values, figure=zooming_in_fig,
                        **self.scalar_map_2d_kwargs)
        traj_handles = plot_trajectories(self.points, figure=zooming_in_fig, **self.trajectories_kwargs)
        self.trajectory_handles = traj_handles[0::2]
        self.location_handles = traj_handles[1::2]
        zooming_in_fig.x_range = Range1d(*xlim)
        zooming_in_fig.y_range = Range1d(*ylim)
        self.roi_view_zooming_in = zooming_in_fig
        return zooming_in_fig

    def slider(self):
        ctrl = self.roi_controller
        zooming_in_fig = self.roi_view_zooming_in
        slider = Slider(start=1, end=len(self.roi_model), step=1, value=self.first_active_roi+1, **self.slider_kwargs)
        slider.js_on_change('value_throttled', ctrl.js_callback(zooming_in_fig))
        self.roi_view_slider = slider
        return slider

    def visibility_button1(self):
        self.trajectory_visibility_button = Toggle(label='Hide lines', button_type='success')
        assert not self.trajectory_handles[1:]
        def set_visibility(multiline):
            return CustomJS(args=dict(multiline=multiline), code="""
                    multiline.visible=!multiline.visible;
                    if (multiline.visible) {
                        this.label='Hide lines';
                    } else {
                        this.label='Show lines';
                    }
                    """)
        self.trajectory_visibility_button.js_on_click(set_visibility(self.trajectory_handles[0]))
        return self.trajectory_visibility_button

    def visibility_button2(self):
        self.location_visibility_button = Toggle(label='Hide points', button_type='success')
        assert not self.location_handles[1:]
        def set_visibility(multiline):
            return CustomJS(args=dict(multiline=multiline), code="""
                    multiline.visible=!multiline.visible;
                    if (multiline.visible) {
                        this.label='Hide points';
                    } else {
                        this.label='Show points';
                    }
                    """)
        self.location_visibility_button.js_on_click(set_visibility(self.location_handles[0]))
        return self.location_visibility_button

    def make_default_view(self):
        full_fov_map = self.full_fov()
        zooming_in_map = self.zooming_in()
        visibility_button1 = self.visibility_button1()
        visibility_button2 = self.visibility_button2()
        if 1 < len(self.roi_model):
            slider = self.slider()
            self.roi_view = row(zooming_in_map, column(slider, full_fov_map, row(visibility_button1, visibility_button2), sizing_mode='scale_width'))
        else:
            self.roi_view = row(zooming_in_map, column(full_fov_map, sizing_mode='scale_width'))

    def show(self):
        show(self.roi_view)


def roi_plot(roi, *args, **kwargs):
    from tramway.helper import map_plot
    from .map import scalar_map_2d, plot_points

    feature_name = kwargs.pop('feature')

    kwargs.update(dict(use_bokeh=True, show=False))
    if 'point_style' not in kwargs:
        kwargs['point_style'] = dict(color='r', alpha=.1)
    fig = map_plot(*args, **kwargs)
    full_fov_fig = fig[0]

    patches_x, patches_y = [], []
    for center, bounding_box in roi:
        patch_x = bounding_box[[0,0,2,2]]
        patch_y = bounding_box[[1,3,3,1]]
        patches_x.append(patch_x)
        patches_y.append(patch_y)

    roi_controller = RoiController(patches_x, patches_y, fill_color=None)
    full_fov_fig.patches(**roi_controller.patches_kwargs)

    roi_controller.unset_active('all')
    first_roi = 0
    roi_controller.set_active(first_roi)

    # right panel

    first_roi_bb = roi[first_roi][1]
    xlim = first_roi_bb[[0,2]].tolist()
    ylim = first_roi_bb[[1,3]].tolist()

    from bokeh.plotting import figure, show

    analysis_tree = args[0]
    partition_label, map_label = kwargs['label']
    cells = analysis_tree[partition_label].data
    _map = analysis_tree[partition_label][map_label].data[feature_name]

    _kwargs = {}
    for _attr in ('clim',):
        if _attr in kwargs:
            _kwargs[attr] = kwargs[attr]

    focused_fig = figure()
    scalar_map_2d(cells, _map, figure=focused_fig, xlim=xlim, ylim=xlim, **_kwargs)
    plot_points(cells, color='r', alpha=.1)

    from bokeh.models import Slider, CustomJS
    slider = Slider(start=1, end=len(roi), step=1, value=1, title='roi')
    slider.js_on_change('value', roi_controller.js_callback(focused_fig))

    # plot with layout

    from bokeh.layouts import row, column
    show(row(full_fov_fig, column(focused_fig, slider, sizing_mode='scale_width')))

