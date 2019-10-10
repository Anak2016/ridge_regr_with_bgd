import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *
from parameters import *

from preprocessing import *

def centering(x):
    '''

    :param x: numpy data ; dim = n*m where n = # instances and m = * features
    :return:
    '''
    return x - x.astype(float).mean(axis=0)
def standalize(x):
    '''

    :param x:  numpy data ; dim = n*m where n = # instances and m = * features
    :return:
    '''
    return x / x.astype(float).std(axis=0)

# def preprocess_data(x):
#     '''
#
#     :param x:
#     :return: return centering + standalize data
#     '''
#     labels = x[:,-1]
#     mask = np.array([True if not isinstance(i,str) else False for i in x[0,:-1]])# mask for col index that is not categorized
#     # for each col find number of unique value and assign (0,1) to it
#     # create directionary of
#     x[:, mask] = centering(x[:, mask])
#     x[:, mask] = standalize(x[:, mask])
#     inv_mask = list(map(bool, 1 - mask))
#     onehot_vec = create_onehot(x[:, inv_mask], num_categories=2) # replace these columns with
#     # x = np.delete(x, inv_mask, 0) # x[: 1- mask]
#     x = x[:, mask]
#     x = np.hstack((x, onehot_vec))
#     # display2screen()
#     return x

def create_onehot(val, num_categories=2):
    '''

    :param x: n * p where n = number of instance and p = number of features
            feature must be categorical features
    :return: return onehot vector of each categorical features
    '''
    if num_categories == 2:
        for col_ind in range(0,val.shape[1]):
            #TODO here>>
            # make it run faster
            mem_ind = {}
            members = np.unique(val[:,col_ind])
            mem_ind = {mem:i for i,mem in enumerate(members)}
            val[:,col_ind] = np.array([mem_ind[mem] for mem in val[:, col_ind]])
        return val

    if num_categories < 2:
        raise ValueError("num_categories must be at least 2. if less WHY DO YOU NEED TO CREATE ONE HOT ANYWAY???")

    if num_categories > 2:
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(handle_unknown='ignore')
        # X = [['Male', 1], ['Female', 3], ['Female', 2]]
        enc.fit(val)
        return enc.transform(val).todense()

if __name__ == '__main__':
    WORKING_DIR = r"C:\Users\awannaphasch2016\PycharmProjects\ridge_regr_with_bgd"
    tmp = f'{WORKING_DIR}/datasets/Credit_N400_p9.csv'
    print(f"reading data from {tmp}...")
    data = pd.read_csv(tmp, sep=',').to_numpy().astype(object)
    data = preprocess_data(data)
    # display2screen(data)
    #check that data is standalized and centered
    mask = [True if not isinstance(i, str) else False for i in data[0, :]]  # mask for col index that is not categorized
    assert data[:, mask].mean(axis=0).sum() != 0, 'data is not centered '
    assert data[:, mask].astype(float).std(axis=0).sum() != 0 , 'data is not stadalized'
    print("preprocessed data is completed !!")


