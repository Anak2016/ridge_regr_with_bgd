import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

import math
import numpy as np
from sympy import symbols, diff
from arg_parser import *
from models.loss_func import MSE
from models.ridge_regr import RidgeRegression

def glorot(val):
    stdv = math.sqrt(6.0 / (val.shape[-2] + val.shape[-1]))
    return np.random.uniform(-stdv, stdv, size=(val.shape[0], val.shape[1]))

def differentiate(loss_func, loop_num, learned_param_name=None, all_params=None, *args, **kwargs):
    '''differnetiate loss function with respecto var'''
    param = symbols(f'{learned_param_name}')
    all_params = {name:val for name, val in all_params.items()}
    diff_loss_val = diff(loss_func.run(loop_num), f"{param}").subs([(name, val) for name, val in all_params.items()])

    return diff_loss_val

    # return loss_func.take_derivative(loop_num, learned_param=learned_param)
#
# def square_matrix_element_wise(x):
#     ''' change value in place'''
#     for i, val in enumerate(x):
#         x.row_op(i, lambda v, j: v*v)
#     return x

# def cross_validation(x, labels, cv, model):
def cross_validation(x, labels, cv):
    '''

    :param x:
    :param labels:
    :param cv:
    :param model:
    :return: output of mean of all cv runs
    '''
    test_size = x.shape[0]/cv
    s = 0
    f = test_size

    loss_val_dict = {}
    loss_val_per_cv = []
    lmda_list = [0.01, 0.1, 1, 10, 100, 1000,10000]

    for lmda in lmda_list:
        args.lmda = lmda
        for i in range(0,cv):
            model = RidgeRegression(x=x,
                                    labels=labels,
                                    loss_func=MSE,
                                    batch_size=args.bs,
                                    learning_rate=args.lr,
                                    epochs=args.epochs)

            s = s + i * test_size
            f = s + test_size if s + test_size > x.shape[0] + 1 else x.shape[0] + 1
            test_mask = [True if (i < f and i >=f) else False for i in range(0,x.shape[0]) ]
            train_mask = [not i for i in test_mask]
            x_train, y_train, x_test, y_test = x[train_mask], labels[train_mask], x[test_mask], labels[test_mask]

            #--------trian
            model.train(x_train, y_train, i)
            #--------predict
            loss_val = model.pred(x_test, y_test, i)
            loss_val_per_cv.append(loss_val)
            # loss_val_cv.setdefault('per_cv', []).append(loss_val)

        # cv_mean = mean(loss_val_per_cv)

        cv_mean = sum(loss_val_per_cv) / len(loss_val_per_cv)
        loss_val_dict.setdefault('per_run', []).append(cv_mean)

        # if lmda == 0.1:
        #     break
    try:
        model.visualize_cv_error_vs_lmda(loss_val_dict['per_run'], lmda_list, save_path = '../output/plot/visualize_cv_error_vs_lmda.png')
    except:
        model.visualize_cv_error_vs_lmda(loss_val_dict['per_run'], lmda_list, save_path='visualize_cv_error_vs_lmda.png')