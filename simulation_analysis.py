"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only. 
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify, 
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights. 
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability, 
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses. 
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from __future__ import print_function
import matplotlib
import matplotlib.pylab as plt
import adafdr.method as md
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import pickle
import sys
import time
import os
import torch
import torch.nn as nn
from two_groups_beta import BlackBoxTwoGroupsModel, M_BlackBoxTwoGroupsModel, M2_BlackBoxTwoGroupsModel
from utils import p_value_2sided, bh_predictions


def get_fdp_and_power(h, h_hat):
    fdp = np.sum((h == 0) & (h_hat == 1)) / np.sum(h_hat == 1)
    power = np.sum((h == 1) & (h_hat == 1)) / np.sum(h == 1)
    return fdp, power


def main(args):
    # Set up the parameters.
    input_folder = args.input_folder
    output_folder = './temp_' + args.data_name + '_ws/res_' + args.data_name
    if args.alpha is not None:
        alpha_list = [args.alpha]
    else:
        alpha_list = [0.05, 0.1, 0.15, 0.2]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        filelist = [os.remove(os.path.join(output_folder, f))\
                    for f in os.listdir(output_folder)]
    print('input_folder: %s' % input_folder)
    print('output_folder: %s' % output_folder)
    print('alpha_list: %s' % alpha_list)

    epochs = 100
    folds = 5

    # Get a file for recording.
    f_write = open(output_folder + '/result.log', 'w')
    # Process all data in the folder
    file_list = os.listdir(args.input_folder)
    result_dic = {
        'bh': [],
        'sbh': [],
        'bbfdr': [],
        'mbbfdr': [],
        'neurtfdra': [],
        'neurtfdrb': [],
        'adafdr': []
    }
    for filename in file_list:
        if filename[0] == '.':
            continue
        file_path = args.input_folder + '/' + filename

        dat = np.load(file_path)
        x1 = dat[:, 0:100]
        x2 = dat[:, 100:105]
        h = np.concatenate(dat[:, 105:106])
        z = np.concatenate(dat[:, 106:107]).clip(-10, 10)

        for alpha in alpha_list:
            print('# Processing %s with alpha=%0.2f' % (filename, alpha))
            f_write.write('# Processing %s with alpha=%0.2f\n' %
                          (filename, alpha))

            # BH result
            bh_preds = bh_predictions(p_value_2sided(z), alpha)
            fdp, power = get_fdp_and_power(h, bh_preds)
            result_dic['bh'].append([fdp, power, alpha, filename])
            f_write.write('## BH discoveries: %d\n' % (bh_preds.sum()))

            # SBH result
            n_rej, t_rej, pi0_hat = md.sbh_test(p_value_2sided(z),
                                                alpha=alpha,
                                                verbose=False)
            fdp, power = get_fdp_and_power(h, p_value_2sided(z) <= t_rej)
            result_dic['sbh'].append([fdp, power, alpha, filename])
            temp = '## SBH discoveries: %d\n' % (n_rej)
            f_write.write(temp)

            # BB-FDR result
            start_time = time.time()
            fdr_model = BlackBoxTwoGroupsModel(x1, z, alpha)
            print('Training for BB-FDR')
            sys.stdout.flush()
            results = fdr_model.train(
                save_dir=output_folder + 'BB-FDR_{}_twogroups'.format(alpha),
                verbose=True,
                batch_size=60 if x1.shape[0] > 1000 else 10,
                num_folds=folds,
                num_epochs=epochs)

            h_predictions = results['predictions']
            fdp, power = get_fdp_and_power(h, h_predictions)
            result_dic['bbfdr'].append([fdp, power, alpha, filename])
            f_write.write('## BB-FDR discoveries: %d\n' %
                          (h_predictions.sum()))
            f_write.write('## Time: %0.1fs' % (time.time() - start_time))

            # MBB-FDR result
            start_time = time.time()
            x = pd.concat(
                [pd.DataFrame(x1), pd.DataFrame(x2)], axis=1).to_numpy()
            fdr_model = BlackBoxTwoGroupsModel(x, z, alpha)
            print('Training for MBB-FDR')
            sys.stdout.flush()
            results = fdr_model.train(
                save_dir=output_folder + 'MBB-FDR_{}_twogroups'.format(alpha),
                verbose=True,
                batch_size=60 if x.shape[0] > 1000 else 10,
                num_folds=folds,
                num_epochs=epochs)

            h_predictions = results['predictions']
            fdp, power = get_fdp_and_power(h, h_predictions)
            result_dic['mbbfdr'].append([fdp, power, alpha, filename])
            f_write.write('## MBB-FDR discoveries: %d\n' %
                          (h_predictions.sum()))
            f_write.write('## Time: %0.1fs' % (time.time() - start_time))

            # NeurT-FDR result A
            start_time = time.time()
            fdr_model = M2_BlackBoxTwoGroupsModel(x1, x2, z, alpha)
            print('Training for NeurT-FDR a')
            sys.stdout.flush()
            results = fdr_model.train(
                save_dir=output_folder +
                'NeurT-FDRa_{}_twogroups'.format(alpha),
                verbose=True,
                batch_size=60 if x1.shape[0] > 1000 else 10,
                num_folds=folds,
                num_epochs=epochs)

            h_predictions = results['predictions']
            fdp, power = get_fdp_and_power(h, h_predictions)
            result_dic['neurtfdra'].append([fdp, power, alpha, filename])
            f_write.write('## NeurT-FDRa discoveries: %d\n' %
                          (h_predictions.sum()))
            f_write.write('## Time: %0.1fs' % (time.time() - start_time))

            # NeurT-FDR result B
            start_time = time.time()
            x = pd.concat(
                [pd.DataFrame(x1), pd.DataFrame(x2)], axis=1).to_numpy()
            fdr_model = M2_BlackBoxTwoGroupsModel(x, x2, z, alpha)
            print('Training for NeurT-FDR b')
            sys.stdout.flush()
            results = fdr_model.train(
                save_dir=output_folder +
                'NeurT-FDRb_{}_twogroups'.format(alpha),
                verbose=True,
                batch_size=60 if x.shape[0] > 1000 else 10,
                num_folds=folds,
                num_epochs=epochs)

            h_predictions = results['predictions']
            fdp, power = get_fdp_and_power(h, h_predictions)
            result_dic['neurtfdrb'].append([fdp, power, alpha, filename])
            f_write.write('## NeurT-FDRb discoveries: %d\n' %
                          (h_predictions.sum()))
            f_write.write('## Time: %0.1fs' % (time.time() - start_time))

            # AdaFDR-fast result
            start_time = time.time()
            p = p_value_2sided(z)
            res = md.adafdr_test(p, x2, alpha=alpha, fast_mode=True)

            n_rej = res['n_rej']
            t_rej = res['threshold']
            fdp, power = get_fdp_and_power(h, p <= t_rej)
            result_dic['adafdr'].append([fdp, power, alpha, filename])
            temp = '## AdaFDR-fast discoveries: fold_1=%d, fold_2=%d, total=%d\n'%\
                  (n_rej[0],n_rej[1],n_rej[0]+n_rej[1])
            f_write.write(temp)
            f_write.write('## Time: %0.1fs' % (time.time() - start_time))

            f_write.write('\n')
    # Store the result
    fil = open(output_folder + '/result.pickle', 'wb')
    pickle.dump(result_dic, fil)
    fil.close()
    f_write.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Side-info assisted multiple hypothesis testing')
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-d', '--data_name', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=float, required=False)
    args = parser.parse_args()
    main(args)
