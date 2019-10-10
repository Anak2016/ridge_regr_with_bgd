import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *
sys.path.insert(1,f'{USER}\\PycharmProjects\\ridge_regr_with_bgd')
# from models.utils import glorot, differentiate, cross_validation
# from models.loss_func import mean_square_error
from models.loss_func import MSE
from arg_parser import *
from data import CreditData
import math
from preprocessing import *

def glorot(val):
    stdv = math.sqrt(6.0 / (val.shape[-2] + val.shape[-1]))
    return np.random.uniform(-stdv, stdv, size=(val.shape[0], val.shape[1]))

# def differentiate(loss_func, loop_num, learned_param_name=None, all_params=None, *args, **kwargs):
#     '''differnetiate loss function with respecto var'''
#     param = symbols(f'{learned_param_name}')
#     all_params = {name:val for name, val in all_params.items()}
#     diff_loss_val = diff(loss_func.run(loop_num), f"{param}").subs([(name, val) for name, val in all_params.items()])
#
#     return diff_loss_val
def visualize_beta_coff_vs_lamda(all_beta_coeffvslamda, save_path):
    print(all_beta_coeffvslamda)
    for i in all_beta_coeffvslamda.values():
        # i = [i for i in i[0]]
        i = i[:6]
        plt.plot(["0.01", "0.1", '1', "10", "100", "1000"], i)
        plt.xscale("log")
        # plt.plot(range(len(self.beta_coeff[f'beta_coeff{i}'])), self.beta_coeff[f'beta_coeff{i}'])
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)

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
    # lmda_list = [0.01, 0.1, 1, 10, 100, 1000,10000]
    lmda_list = [0.01, 0.1, 1, 10, 100, 1000,10000]
    # lmda_list = [1]
    beta_coeff = {}
    for lmda in lmda_list:
        args.lmda = lmda
        model = RidgeRegression(x=x,
                                labels=labels,
                                loss_func=MSE,
                                batch_size=args.bs,
                                learning_rate=args.lr,
                                epochs=args.epochs)
        for i in range(0,cv):
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
        # --------get beta_coeff of last epoch (converged epoch)
        beta_coeff.setdefault(f'{lmda}',[v[-1] for k,v in model.get_beta_coeff().items()]).append([v[-1] for k,v in model.get_beta_coeff().items()])

        # cv_mean = mean(loss_val_per_cv)
        cv_mean = sum(loss_val_per_cv) / len(loss_val_per_cv)
        loss_val_dict.setdefault('per_run', []).append(cv_mean)

        # model.visualize_beta_coff(f'C:\\Users\\Anak\\PycharmProjects\\ridge_regr_with_bgd\\output\\plot\\beta_coeffvs_lmda={args.lmda}.png') # use beta_coff of the last cv run

    #TODO here>>
    visualize_beta_coff_vs_lamda(beta_coeff, f'C:\\Users\\Anak\\PycharmProjects\\ridge_regr_with_bgd\\output\\plot\\all_beta_coeffvs_lmda.png') # use beta_coff of the last cv run

    #TODO here>>
    try:
        print(loss_val_dict['per_run'])
        model.visualize_cv_error_vs_lmda(loss_val_dict['per_run'], lmda_list, save_path = '../output/plot/visualize_cv_error_vs_lmda.png')
    except:
        model.visualize_cv_error_vs_lmda(loss_val_dict['per_run'], lmda_list, save_path='visualize_cv_error_vs_lmda.png')

class MessagePassing():
    def __init__(self, x, y, loss_func=None, batch_size=None, epochs=None, *args, **kwargs):
        self.register_params = {}
        self.y = y # labels
        self.loss_func = loss_func # loss_func
        self.bs = batch_size
        self.data = x
        self.epochs = epochs
        self.loss_val_hist = []
        self.loss_val = None
        self.beta_coeff = {}
        self.__args__ = args
        self.__kwargs__ = kwargs
        # display2screen(len(self.__args__), self.__kwargs__)

    def register(self, **kwargs):
        '''register variable to be used in differentiate()'''
        for i,(k,v) in enumerate(kwargs.items()):
            self.register_params[f'var_{i}'] = {'name': k,'val':v, 'diff_val': None} # is this the best datastrcuture to be used?
        # display2screen(self.register_params)

    # def get_loss_val(self, loop_num, all_params=None):
    #     all_params = {name: val for name, val in all_params.items()}
    #     #TODO here>> check if loss_val is correct
    #     # > consider using given formular
    #
    #     # loss_val = self.loss_func.run(loop_num).subs([(name, val) for name, val in all_params.items()])
    #     loss_val = (self.data.T.dot(self.y - self.data.dot(v['val'])) + 2 * args.lmda * v['val'])
    #     return loss_val

    def step(self, val, diff_val, epoch_num, loss_diff_percent):
        # return val + diff_val * args.lr
        # if epoch_num ==0:
        #     args.lr = 0.0009

        # # if epoch_num!=0 and epoch_num % 100 == 0 :
        # if loss_diff_percent is not None:
        #     if  loss_diff_percent < 0.001:
        #         args.lr = args.lr * 10
        args.lr = 0.00009

        return val - diff_val * args.lr

    def backward_batch(self, x,y ,loop_num, epoch_num, cv_num):
        self.data = x
        self.y = y

        i = loop_num
        all_params = {f"{v['name']}{i}" : {"val":v['val'][i], "diff_val":None} for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}
        # print(f"epoch_num={epoch_num} loop={loop_num}; val= {self.register_params['var_0']['val']}")
        all_params_no_diff_val = {f"{v['name']}{i}" : v['val'][i]for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}
        loss_diff_percent = None
        try:
            #--------variable within function
            for k, v in self.register_params.items():
                if k.startswith('var_'):
                    reconstruct_diff_vals = []
                    #TODO here>> create an example and expected output of this example
                    # v['val'] = np.array([2,3,7])
                    # self.data = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
                    # args.lmda = 0.1
                    # self.y = np.array([11,22,33, 44])

                    # if loop_num == 0 and epoch_num ==0:
                    #     self.data = self.data
                    #     # self.data = self.data[:2,:-3]
                    #     self.y = self.y
                    #     # v['val'] = v['val'][:-3]
                    #     v['val'] = v['val']
                    #     # args.lmda = 0.01
                    #     args.lmda = 0.01

                    #TODO here>> check calculation for loss_val
                    # > check how
                    if epoch_num == 20:
                        print('hi')

                    #TODO here>> early stop; epoch_num = 12 learning rate = 0.0009 and lmda = 100
                    loss_val = ((self.y - self.data.dot(v['val'])) ** 2).sum(axis=0) +  args.lmda * (v['val'] ** 2).sum() # result too large
                    self.loss_val = loss_val
                    v['diff_val'] = (- 2 * (self.data.T.dot(self.y - self.data.dot(v['val']))) + (2 * args.lmda * v['val'])) # decrease iteratively
                    self.pred_val = self.data.dot(v['val'])
                    # print(f"diff_val = {v['diff_val']}")

                    print(f"first_term = {((self.y - self.data.dot(v['val'])) ** 2).sum(axis=0)} ")
                    print(f"second_term = {args.lmda * (v['val'] ** 2).sum()}")
                    print(f"learning rate = {args.lr} and lmda = {args.lmda}")

                    self.loss_val_hist.append(loss_val)
                    if epoch_num >= 1 or loop_num >= 1:
                        loss_diff_percent = (self.loss_val_hist[-2] - self.loss_val_hist[-1]) / self.loss_val_hist[-2]
                        print(f"loss_diff = {self.loss_val_hist[-1] - self.loss_val_hist[-2]}")

                    # v['diff_val'] =   2 * (self.data * (self.y - np.expand_dims(self.data.dot(v["val"]), axis=1)) + 2 * args.lmda * v["val"].squeeze()).sum(axis =0)
                    # display2screen(f"diff_val = {v['diff_val']}", f"loss_val = {loss_val}") # this suppose to have dim = 3
                    self.register_params[k]['val'] = self.step(v['val'], v['diff_val'], epoch_num, loss_diff_percent)  # update beta value
                    # print()

            if epoch_num == 1:
                pass

            # loss_val = np.array(self.get_loss_val(loop_num, all_params = all_params_no_diff_val)).squeeze()

            if args.verbose:
                print(f'cv={cv_num}, epoch={epoch_num}, batch={self.bs}'
                      f'    ==> {i * self.bs}: loss_val={loss_val}')
        except OverflowError:
            self.register_params[k]['val'] = self.previous_val
            raise OverflowError
            # raise OverflowError("reassign beta to value before the error ...")

            #TODO here>> why does not the error propagate to the parants' function
            # print(f'reassign beta to value before the error ...')



    # def backward_batch(self, loop_num, epoch_num, cv_num):
    #     i = loop_num
    #     all_params = {f"{v['name']}{i}" : {"val":v['val'][i], "diff_val":None} for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}
    #     # print(f"epoch_num={epoch_num} loop={loop_num}; val= {self.register_params['var_0']['val']}")
    #     all_params_no_diff_val = {f"{v['name']}{i}" : v['val'][i]for k,v in self.register_params.items() for i in range(0,v['val'].shape[0])}
    #
    #     for k, v in self.register_params.items():
    #         if k.startswith('var_'):
    #             reconstruct_diff_vals = []
    #             for param,value in all_params_no_diff_val.items():
    #                 all_params[param]['diff_val'] = differentiate(self.loss_func, loop_num, learned_param_name = param, all_params = all_params_no_diff_val)
    #                 reconstruct_diff_vals.append(all_params[param]['diff_val'])
    #
    #             array_diff_val = np.array(reconstruct_diff_vals)
    #             # print(array_diff_vals)
    #             self.register_params[k]['diff_val'] =  array_diff_val.squeeze()
    #
    #             # TODO here>> construct diff_val from beta0 to beta10 as array
    #             #  > look at plot of loss_val vs different lambda (penality coefficient)
    #             #  > look at plot of loss_val vs leraning rate
    #             #  > try to speed up the derivative  process by skipping derivative step
    #             #  > implement cross validation
    #             #  > plot loss val vs epoch
    #             #  > plot beta vs lamda (penality coefficient)
    #             #  > why loss_val increse each run?
    #             #        >> why is there always a jump of loss_val between epochs
    #             #        >> why is self.register_params[k]['val'] repeat same score sequence from batch after certain epoch??
    #             #        >> diff_val is alot higher than val after derivation
    #             #  > check why loss_val is not steadily decreasing; do i step in the right direction??
    #             #  > run backward_batch for many
    #             #  > cross validation
    #             #  > plot loss_val over time
    #             self.register_params[k]['val'] = self.step(v['val'], v['diff_val'])  # update beta value
    #     if epoch_num == 1:
    #         pass
    #     loss_val = np.array(self.get_loss_val(loop_num, all_params = all_params_no_diff_val)).squeeze()
    #     # print(f"labels = {self.y[:10]} prediction = {(self.data.dot(self.register_params[k]['val']))[:10]}")
    #     print(f"distance = {(self.y - self.data.dot(self.register_params[k]['val'])).sum()}")
    #     self.loss_val_hist.append(loss_val)
    #     if args.verbose:
    #         print(f'cv={cv_num}, epoch={epoch_num}, batch={self.bs} index '
    #               f'    ==> {i * self.bs}: loss_val={loss_val}')



    def backward(self, x,y ,epoch, cv_num, beta):
        '''
        does backpropagation
        :return: all data

        '''
        print('doing backward..')
        assert len(self.register_params) > 0, "no param has been registed "
        batch_loop = 1 + int(self.data.shape[0]/ self.bs) if self.data.shape[0]/ self.bs   != int(self.data.shape[0]/ self.bs) + 1 else int(self.data.shape[0])
        try:
            for i in range(0, batch_loop ):
                self.backward_batch(x, y, i, epoch, cv_num)
        except OverflowError:
            raise

        # --------beta coeff
        for k,v in self.register_params.items():
            if k.startswith('var_'):
                for i in range(v['val'].shape[0]):
                    self.beta_coeff.setdefault(f'beta_coeff{i}', []).append(v['val'][i])

        # display2screen(self.register_params.items())
        self.previous_val = [v['val'] for k,v in self.register_params.items()][0] # previous val to be trace back

        result = {v['name']: v['val'] for k,v in self.register_params.items()}
        result.update({"pred_val":self.pred_val, "beta_coeff": self.beta_coeff, "loss_val":self.loss_val})
        return result


class RidgeRegression(MessagePassing):
    ''' y = beta*X + intercept where b is intersect and A i slope of hyperplan'''
    def __init__(self, x=None,labels=None, loss_func=None, batch_size=None, learning_rate = None, epochs=None):
        '''

        :param data:
        :param loss_func: a class of loss function (not function or method)
        :param batch_size:
        :param learning_rate:
        '''
        # self.y = np.zeros((data.shape[0], 1)) # prediction
        self.y = labels
        self.x = x
        self.bs = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.loss_val_cv = None

        # --------message_passing
        # self.message_passing = MessagePassing()
        self._initialize_parameters()
        self.loss_func = None
        # self.loss_func = loss_func( self.y,self.x,self.beta, lmda=args.lmda, l1=True, batch_size=self.bs)
        super(RidgeRegression, self).__init__(self.x, self.y, self.loss_func, batch_size, epochs)

        # self.message_passing.register('a', self.beta) # register beta as a message
        self.register(beta= self.beta.squeeze()) # register beta as a message
        # self.register(intercept=self.intercept)
    def get_beta_coeff(self):
        return self.beta_coeff

    def _initialize_parameters(self):
        # self.beta = np.random.uniform((data.shape[1],))
        # self.beta = np.zeros((self.x.shape[0],self.x.shape[1],))
        #TODO here>> here fix beta dim
        if x.shape[1] != 10:
            self.x = np.hstack((self.x,np.ones((self.x.shape[0],1))))

        assert x.shape[1] == 10,"x.shape[1] == 10"

        self.beta = np.zeros((1,self.x.shape[1]))
        self.beta = glorot(self.beta)

        # display2screen(self.beta.shape, self.x.shape)

        # self.intercept = np.zeros((self.x.shape[0]))
        # self.intercept = glorot(self.beta)

    def train(self, x ,y, cv_num):
        self.x, self.y = x, y
        self._initialize_parameters()

        epoch = None
        #TODO here>> how to do try catch for this case???
        try:
            for epoch_num in range(0,self.epochs):
                # get cv error here
                epoch = epoch_num
                # ridg_regr.run(self.x, self.y, epoch_num, cv_num)
                self.run(self.x, self.y, epoch_num, cv_num)
                if args.report_performance:
                    pass
                    # report_performances()
        except OverflowError as e:
            raise
            # print(f'''#=====================
            # == early stop; epoch_num = {epoch} learning rate = {args.lr} and lmda = {args.lmda}
            # =====================''')
            # print('caught error ')

            # print(f"learning rate = {args.lr} and lmda = {args.lmda}")
            # display2screen('here')

        if args.report_performance:
            pass
            # report_performances()

    def pred(self, x, y, cv):
        '''
        predict performance from test set
        :return:
        '''
        # self.x, self.y = x, y
        print('========================')
        print('========================')
        print(f"predicting cv = {cv}") #
        print(f'beta = {self.beta}')
        print(f'beta_coeff = {self.beta_coeff}')

        #--------prediciting
        # ridg_regr.run(self.x, self.y, epoch_num=0, cv_num=0)
        self.run(self.x, self.y, epoch_num=0, cv_num=0)
        # self.loss_val_cv.setdefault('loss_val_cv', []).append(self.loss_val)

        #--------compute accuracy
        
        if args.report_performance:
            pass
            # report_performances()

        return self.loss_val

    def apply_formular(self,x):
        return self.x.dot(self.beta.T)

    def run(self, x, y, epoch_num, cv_num):
        # self.forward()
        #TODO here>> check that beta is updated correctly after backward is completely.
        try:
            self.result = self.backward(x, y, epoch_num, cv_num, self.beta)
        except OverflowError:
            raise
        self.update(self.result) # update value of beta


    def update(self, params_dict=None):
        # self.beta = [j for i, j in params_dict.items() if i == 'beta'][0]  # get beta value
        for i, j in params_dict.items():
            if i == 'beta':
                self.beta = j
            if i == 'pred_val':
                self.pred_val = j
            if i == 'beta_coeff':
                self.beta_coeff = j
            if i == 'loss_val':
                self.loss_val = j

    # def forward(self):
    #     print('doing forward..')
    #     #TODO here>> update beta in RidgeRegression
    #     # > directly compare prediction after 1 epoch. ( accuracy)
    #     self.y = self.x.dot(self.beta.T)
    #     # return self.x.dot(self.beta.T) + self.intercept  # result #

    def visualize_result(self):
        mask = np.argsort(self.y)
        plt.plot(range(self.y.shape[0]), self.y[mask], 'go--')
        plt.plot(range(self.pred_val.shape[0]), self.pred_val[mask])
        plt.show()


    def visualize_beta_coff(self,save_path=None):
        '''
        x = lamda
        y = cv error
        :return:
        '''
        for i in range(len(self.beta_coeff)):
            plt.plot(range(len(self.beta_coeff[f'beta_coeff{i}'])), self.beta_coeff[f'beta_coeff{i}'])
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def visualize_cv_error_vs_lmda(self, cv_error, lmda, save_path = None):
        '''
        :param cv_error:  list of mean of 5 fold cv_error for all different lmda value
        :param lmda: list of lamda value
        :return:
        '''
        # xticks = list(map(str,lmda))
        # print(cv_error)
        objects = list(map(str, lmda))
        y_pos = np.arange(len(objects))
        performance = cv_error

        plt.bar(y_pos, performance, align='center', alpha=0.5)
        # plt.xticks(list(range(len(lmda))),['0.01', '0.1'] )
        plt.xticks(y_pos, objects)
        plt.ylabel('cv_error')
        plt.title("cv_error vs lmda")
        plt.ylim(0, 7.5e7 )
        plt.yscale("log")
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)


def run_submission():
    # --------check if bias is added for x
    print("preprocessed data is completed !!")

    # def mean_square_error(y, x, beta, lmda=0.1, l1=True):
    # ridg_regr = RidgeRegression(data, MSE, 32, 0.1, epochs=args.epoch)
    # def __init__(self, data=None, loss_func=None, batch_size=None, learning_rate = None):


    # cross_validation(x, labels, args.cv, ridg_regr)
    # cross_validation(x, labels, args.cv)
    #=====================
    #==predict with best parameters
    #=====================

    ridg_regr = RidgeRegression(x=x,
                                labels=labels,
                                loss_func=MSE,
                                batch_size=32,
                                learning_rate=0.00009,
                                epochs=1000)
    args.lmda= 0.01
    ridg_regr.train(x, labels, 0) # loss value is shown on the last printed line

    # ridg_regr.pred(x, labels, 0)

    # ridg_regr.visualize_result()
    # ridg_regr.visualize_beta_coff()

    # epoch should be done here.
    # ridg_regr.run()
    # --------cross validation
    # cv = args.cv
    # test_size = x.size[0]/cv
    # tmp = [True for i in range(0,cv)]
    # s = 0
    # f = test_size
    # for i in range(0,cv):
    #     s = s + i * test_size
    #     f = s + test_size if s + test_size > x.size[0] + 1 else x.size[0] + 1
    #     test_mask = [True if (i < f and i >=f) else False for i in range(0,x.size[0]) ]
    #     train_mask = [not i for i in test_mask]
    #     x_train, y_train, x_test, y_test = x[train_mask], labels[train_mask], x[test_mask], labels[test_mask]
    #     #--------trian
    #     ridg_regr.train(x_train, labels.trian)
    #     #--------predict
    #     ridg_regr.predict(x_test, labels.test)


if __name__ == '__main__':

    WORKING_DIR = r"C:\Users\awannaphasch2016\PycharmProjects\ridge_regr_with_bgd"
    tmp = f'{USER}/PycharmProjects/ridge_regr_with_bgd/datasets/Credit_N400_p9.csv'
    print(f"reading data from {tmp}...")
    data = pd.read_csv(tmp, sep=',').to_numpy().astype(object)[:, 1:]
    credit_data = CreditData(data)
    credit_data.preprocess_data()
    x = credit_data.x
    labels = credit_data.y
    # display2screen(data)
    # check that data is standalized and centered
    # mask = [True if not isinstance(i, str) else False for i in x[0, :]]  # mask for col index that is not categorized
    mask = credit_data.mask
    # assert x[:, mask].mean(axis=0).sum() != 0, 'data is not centered '
    # assert x[:, mask].astype(float).std(axis=0).sum() != 0, 'data is not stadalized'
    assert x[:, :-1][:, mask].mean(axis=0).sum() != 0, 'data is not centered '
    assert x[:, :-1][:, mask].astype(float).std(axis=0).sum() != 0, 'data is not stadalized'
    assert x.shape[1] == 10, 'bias must be added as 10th dim of features'
    run_submission()