from preprocessing import centering, standalize, create_onehot
import numpy as np
class CreditData:
    def __init__(self, data):
        self.data= data
        self._x = None
        self._y = None
        self.mask = None
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

    def preprocess_data(self):
        '''

        :paramdata x:
        :return: return centering + standalize data
        '''
        data = self.data
        labels = data[:, -1]
        x = data[:,:-1]
        mask = np.array([True if not isinstance(i, str) else False for i in x[0, :]])  # mask for col index that is not categorized

        # for each col find number of unique value and assign (0,1) to it
        # create directionary of
        x[:, mask] = centering(x[:, mask])
        x[:, mask] = standalize(x[:, mask])
        inv_mask = list(map(bool, 1 - mask))
        onehot_vec = create_onehot(x[:, inv_mask], num_categories=2)  # replace these columns with
        # x = np.delete(x, inv_mask, 0) # x[: 1- mask]
        x[:,inv_mask] = onehot_vec
        #--------add bias
        x = np.hstack((np.expand_dims(np.ones_like(x[:,0]), axis=1), x))

        # print(x.shape)
        # exit()

        # x = data[:, mask]
        # x = np.hstack((x, onehot_vec))

        # display2screen()
        self.x = x
        self.y = labels
        self.mask = mask
