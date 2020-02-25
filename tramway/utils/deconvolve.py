# -*- coding: utf-8 -*-

# Copyright © 2020, Institut Pasteur
#   Contributor: François Laurent

# This file is part of the TRamWAy software available at
# "https://github.com/DecBayComp/TRamWAy" and is distributed under
# the terms of the CeCILL license as circulated at the following URL
# "http://www.cecill.info/licenses.en.html".

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.


import tramway.deconvolution.inference as deconv
import os.path


class _Files(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            attr = key[:-5] if key.endswith('_file') else key
            path = os.path.abspath(val) if isinstance(val, str) else val
            setattr(self, attr, path)


def main(image_stack_file, weight_file, mean_std_file,
        high_res_image_file=None, save_magnified_image=False,
        magnification=10, threshold=0, min_distance_peak=2, margin=3,
        header=True, abs_threshold=1., M=64, N=None, n=2, gpu=1):

    if N is None:
        N = M
    files = _Files(
            img_stack=image_stack_file,
            weights=weight_file,
            mean_std=mean_std_file,
            high_res_img=high_res_image_file,
            )

    mean_img, std_img = deconv.get_parameters_mean_std(files.mean_std)

    if save_magnified_image:
        deconv.save_trimmed_original_image_magnified_for_testing_purposes(
                files.img_stack, magnification, n)

    high_res_prediction, pos = deconv.Inference(files.img_stack,
            magnification, files.weights, mean_img, std_img, M, N, n,
            threshold, min_distance_peak, abs_threshold, margin, 1<gpu)

    [basedir, filename] = os.path.split(files.img_stack)
    basename,_ = os.path.splitext(filename)
    deconv.print_position_files(pos, basedir, basename, header)

    if files.high_res_img:
        if not isinstance(files.high_res_img, str):
            files.high_res_img = os.path.join(basedir, 'predicted.tiff')
        deconv.imsave(files.high_res_img, high_res_prediction.astype('uint16'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stack',    help="path to the tiff image stack")
    parser.add_argument('--weights',  help="path to the weight file")
    parser.add_argument('--mean-std', help="path to the mean_std.txt file")
    parser.add_argument('-n', '--gpu',   type=int, help="number of GPUs")
    args   = parser.parse_args()

    main(args.stack, args.weights, args.mean_std, gpu=args.gpu)

