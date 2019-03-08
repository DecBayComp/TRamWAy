#!/usr/bin/env python


from tramway.helper.animation import animate_map_2d_helper


def mapviz_utility():
    import argparse
    parser = argparse.ArgumentParser(prog='mapviz',
        description='visualize animated 2D maps')
    parser.add_argument('-l', '-L', '--label', help="comma-separated list of labels")
    parser.add_argument('-f', '--feature', help="mapped feature to render")
    parser.add_argument('-r', '--frame-rate', '--fps', default=1, type=float, help="frames per second")
    parser.add_argument('-c', '--codec', help="the codec to use")
    parser.add_argument('-b', '--bit-rate', '--bitrate', type=int, help="movie bitrate")
    parser.add_argument('--dpi', type=int, help="dots per inch")
    parser.add_argument('-q', '--quiet', action='store_true', default=False, help="do not print anything")
    parser.add_argument('--write-only', action='store_true', default=False, help="do not play the movie")
    parser.add_argument('-s', '--time-step', type=float, help="time step between successive frames")
    parser.add_argument('-u', '--time-unit', type=str, default='s', help="time unit for time display")
    parser.add_argument('--colormap', help="matplotlib colormap name")
    parser.add_argument('--axis-off', '--axes-off', action='store_true', help="turn the axes off")
    parser.add_argument('--colorbar-off', action='store_true', help="turn the colorbar off")
    parser.add_argument('--bounding-box', help="bounding box as left,bottom,right,top")
    parser.add_argument('input_file', help='path to rwa file')
    parser.add_argument('output_file', nargs='?', help='path to mp4 file')
    args = parser.parse_args()

    if args.bounding_box:
        bounding_box = Bbox.from_extents(float(x) for x in args.bounding_box.split(','))
    else:
        bounding_box = None

    animate_map_2d_helper(args.input_file, args.output_file,
            label=args.label,
            feature=args.feature,
            frame_rate=args.frame_rate,
            codec=args.codec,
            bit_rate=args.bit_rate,
            play=not args.write_only,
            time_step=args.time_step,
            time_unit=args.time_unit,
            colormap=args.colormap,
            axis=not args.axis_off,
            colorbar=not args.colorbar_off,
            bounding_box=bounding_box,
            verbose=not args.quiet)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    mapviz_utility()

