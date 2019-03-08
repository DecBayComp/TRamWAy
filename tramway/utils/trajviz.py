#!/usr/bin/env python


from tramway.helper.animation import animate_trajectories_2d_helper
from matplotlib.transforms import Bbox


def trajviz_utility():
    import argparse
    parser = argparse.ArgumentParser(prog='trajviz',
        description='visualize animated 2D trajectories')
    parser.add_argument('--columns', help="xyt file column names (comma-separated list)")
    parser.add_argument('-r', '--frame-rate', '--fps', type=float, help="frames per second")
    parser.add_argument('-c', '--codec', help="the codec to use")
    parser.add_argument('-b', '--bit-rate', '--bitrate', type=int, help="movie bitrate")
    parser.add_argument('--dpi', type=int, help="dots per inch")
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help="do not print anything")
    parser.add_argument('--write-only', action='store_true', default=False, help="do not play the movie")
    parser.add_argument('-s', '--time-step', type=float, help="time step between successive frames")
    parser.add_argument('-u', '--time-unit', type=str, default='s', help="time unit for time display")
    parser.add_argument('--line-width', type=float, default=1, help="translocation line width")
    parser.add_argument('--marker-style', type=str, default='o', help="location marker style (can be 'none')")
    parser.add_argument('--marker-size', type=int, default=4, help="location marker size")
    parser.add_argument('--axis-off', '--axes-off', action='store_true', help="turn the axes off")
    parser.add_argument('--bounding-box', help="bounding box as left,bottom,right,top")
    parser.add_argument('input_file', help='path to xyt trajectory file')
    parser.add_argument('output_file', nargs='?', help='path to mp4 file')
    args = parser.parse_args()

    if args.bounding_box:
        bounding_box = Bbox.from_extents(float(x) for x in args.bounding_box.split(','))
    else:
        bounding_box = None

    animate_trajectories_2d_helper(args.input_file, args.output_file,
            frame_rate=args.frame_rate,
            codec=args.codec,
            bit_rate=args.bit_rate,
            dots_per_inch=args.dpi,
            play=not args.write_only,
            time_step=args.time_step,
            time_unit=args.time_unit,
            line_width=args.line_width,
            marker_style=None if args.marker_style == 'none' else args.marker_style,
            marker_size=args.marker_size,
            axis=not args.axis_off,
            bounding_box=bounding_box,
            verbose=not args.quiet,
            columns=args.columns)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    trajviz_utility()

