import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

import numpy as np
from sympy import symbols, diff, Matrix, lambdify, ones, eye, MatrixSymbol
# from models.utils import square_matrix_element_wise

def square_matrix_element_wise(x):
    ''' change value in place'''
    for i, val in enumerate(x):
        x.row_op(i, lambda v, j: v*v)
    return x

class MSE:
    def __init__(self, y,x,beta,lmda=0.1, l1=True, batch_size=None):
        self.y = y
        self.x = x
        self.beta = beta
        self.lmda = lmda
        self.l1 = l1
        self.bs = batch_size


    def run(self, loop_num):
        # check if learned val is one of learnable param in func
        i = loop_num
        s = (i) * self.bs
        f = (i + 1) * self.bs if (i + 1) * self.bs < self.x.shape[0] + 1 else self.x.shape[0] + 1
        #--------create matrixsymbol and pass in value of matrix

        # y = MatrixSymbol('y', self.y[s:f].shape[0],self.y[s:f].shape[1] )
        # x = MatrixSymbol('x', self.x[s:f].shape[0], self.x[s:f].shape[1])
        y = Matrix(self.y[s:f])
        x = Matrix(self.x[s:f])
        beta_sym = symbols(f'beta0:{self.beta.shape[1]}')  # BETA = tuple of length = 1 * p
        beta = Matrix(np.array(beta_sym))

        loss_func = (y - (x * beta)) * ones(beta.shape[1], 1)
        loss_func = ones(1, loss_func.shape[0]) * square_matrix_element_wise(loss_func) / 400 # mean
        # loss_func = ones(1, loss_func.shape[0]) * square_matrix_element_wise(loss_func)  # mean
        # loss_func = np.random.rand(1,1)

        penalty =  self.lmda * (ones(1, beta.shape[0]) * square_matrix_element_wise(beta)) if  self.l1 else 0
        return loss_func + penalty

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
    # def take_derivative(self, loop_num, learned_param = None):
    #     '''return values of loss func wrp val '''
    #
    #     assert learned_param is not None, "learned_param must not be None"
    #     i = loop_num
    #     s = (i) * self.bs
    #     f = (i + 1) * self.bs if (i + 1) * self.bs < self.x.shape[0] + 1 else self.x.shape[0] + 1
    #
    #     learned_param['name'] = learned_param['name'].upper() if learned_param['name'].islower() else learned_param['name']
    #     PRED_VAL = 'PRED_VAL'
    #     DATA = 'DATA'
    #     BETA = 'BETA'
    #     assert learned_param['name'] in [PRED_VAL,DATA,BETA], f"loss_func does not have params named {learned_param['name']}"
    #
    #     params = {PRED_VAL: self.y[s:f],DATA:self.x[s:f],BETA:self.beta}
    #     for k,v in params.items():
    #         #--------bad code; figure out a way to not use exec
    #         # exec(f"{k} = symbols('{k}')")
    #         if k == learned_param['name']:
    #             # for beta, create symbolic value for
    #
    #             if k == BETA:
    #                 # BETA is expected to have dim = p
    #                 beta_sym = symbols(f'BETA0:{params["BETA"].shape[1]}') # BETA = tuple of length = 1 * p
    #                 beta_val =  np.array(beta_sym)
    #                 exec(f"{k} = Matrix(beta_val)") # k.shape = p * 1
    #                 print(f'{learned_param["name"]} = {k}')
    #         else:
    #             exec(f"{k} = Matrix(v)")
    #             print(f'{learned_param["name"]} = {k}')
    #
    #     from models.utils import  square_matrix_element_wise
    #     #TODO here>> replace equation with function
    #     # > first create exmaple test
    #     diff_loss_func = (PRED_VAL - (DATA * BETA)) * ones(BETA.shape[0], 1)
    #     diff_loss_func = square_matrix_element_wise(diff_loss_func) * ones(BETA[diff_loss_func],1)
    #
    #     penalty = self.lmda * square_matrix_element_wise(BETA) * ones(BETA[0], 1) if self.l1 else 0
    #
    #     diff_loss_func = diff_loss_func + penalty
    #
    #     # diff_loss_func = self.loss_func()
    #
    #     return diff(diff_loss_func, learned_param['name']).subs(learned_param['name'], learned_param['val'])

    # def __call__(self, loop_num):
    #     '''
    #
    #     :param y: dim = n
    #     :param x: dim = n * p
    #     :param beta: dim = 1 * p (to automatically broadcast along dim 0)
    #     :param lmda:
    #     :param l1:
    #     :return:
    #     '''
    #     i = loop_num
    #     penalty = 0
    #     if self.l1:
    #         penalty = self.penalty()
    #
    #     # return np.square(self.y - self.x.dot(self.beta.T)).sum(axis=0).squeeze(1) + penalty
    #     s = (i) * self.bs
    #     f = (i + 1) * self.bs if (i + 1) * self.bs < self.x.shape[0] + 1 else self.x.shape[0] + 1
    #
    #
    #     return np.square(self.y[s:f] - self.x[s:f,:].dot(self.beta.T)).sum(axis=0) + penalty

# def mean_square_error(y,x,beta, lmda=0.1, l1=True):
#     '''
#
#     :param y: dim = n
#     :param x: dim = n * p
#     :param beta: dim = 1 * p (to automatically broadcast along dim 0)
#     :param lmda:
#     :param l1:
#     :return:
#     '''
#     penalty = 0
#     if l1:
#         penalty =  lmda * np.square(beta).sum(axis=0)
#
#     return np.square(y - x.dot(beta.T)).sum(axis=1) + penalty