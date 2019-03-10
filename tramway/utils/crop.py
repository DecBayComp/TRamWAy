#!/usr/bin/env python


from tramway.core.xyt import load_xyt, crop
import numpy as np


def cropping_utility():
    import argparse
    parser = argparse.ArgumentParser(prog='crop',
        description='crop 2D trajectories')
    parser.add_argument('--columns', help="input trajectory file column names (comma-separated list)")
    parser.add_argument('-q', '--quiet', action='store_true', help="do not ask to overwrite the file")
    parser.add_argument('input_file', help='path to input trajectory file')
    parser.add_argument('bounding_box', help='bounding box as left,bottom,right,top')
    parser.add_argument('output_file', nargs='?', help='path to output trajectory file')
    args = parser.parse_args()

    bounding_box = np.array([ float(x) for x in args.bounding_box.split(',') ])
    dim = int(round(float(bounding_box.size) * .5))
    bounding_box[dim:] -= bounding_box[:dim]

    load_kwargs = {}
    if args.columns:
        load_kwargs['columns'] = args.columns.split(',')
    trajectories = load_xyt(args.input_file, **load_kwargs)

    cropped_trajectories = crop(trajectories, bounding_box.tolist())

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = args.input_file
        if not args.quiet:
            invite = "overwrite file '{}': [N/y] ".format(output_file)
            try:
                answer = raw_input(invite) # Py2
            except NameError:
                answer = input(invite)
            if not (answer and answer[0].lower() == 'y'):
                return

    cropped_trajectories.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    cropping_utility()

