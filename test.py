'''
Private data Testing 
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
from main import *    


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():

    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()  # Testing_set.csv will be loaded when testing on other device
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    # Won't assign O1 and O2 when testing, make sure the default value is the best choice
    predict_BLR = BLR(data_train, data_test_feature)
    predict_MLR = MLR(data_train, data_test_feature)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label ), e2=CalMSE(predict_MLR, data_test_label)))



if __name__ == '__main__':
    main()
