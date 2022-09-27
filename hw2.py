'''
NTHU EE Machine Learning HW2
Author: 
Student ID: 
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse
import matplotlib.pyplot as plt

def phi(data_feature, O1, O2):
    x1_max = np.amax(data_feature[:, 0])
    x1_min = np.amin(data_feature[:, 0])
    x2_max = np.amax(data_feature[:, 1])
    x2_min = np.amin(data_feature[:, 1])
    
    s1 = (x1_max - x1_min) / (O1 - 1)
    s2 = (x2_max - x2_min) / (O2 - 1)
    
    phi_mat = np.zeros((data_feature.shape[0], (O1 * O2)+2))
    
    for n in range(data_feature.shape[0]):

        for i in range(O1):
            m_i = s1 * i + x1_max;
            
            for j in range(O2):
                m_j = s2 * j + x2_max
                val = math.exp(-((math.pow((data_feature[n, 0]-m_i), 2)/(2*math.pow(s1,2)))+(math.pow((data_feature[n, 1]-m_j), 2)/(2*math.pow(s2,2)))))
                phi_mat[n, i * O2 + j] = val
        
        phi_mat[n, (O1 * O2)] = data_feature[n, 2]
        phi_mat[n, (O1 * O2)+1] = 1
            
    return phi_mat


def MLR_train(train_data, O1, O2):
    #Get training label
    t = train_data[:, 3]
    train_data_feature = train_data[:, :3]
    phi_mat = phi(train_data_feature, O1, O2)   
    weight = np.linalg.inv(phi_mat.T @ phi_mat) @ phi_mat.T @ t
    return weight
    
def BLR_train(train_data, O1, O2):
    t = train_data[:, 3]
    train_data_feature = train_data[:, :3]
    
    alpha = np.sqrt(1/0.5)
    beta = np.sqrt(1/0.5)
    phi_mat = phi(train_data_feature, O1, O2)  
    
    Sn = np.linalg.inv(alpha * np.identity(phi_mat.shape[1]) + beta * phi_mat.T @ phi_mat)
    Mn = beta * Sn @ phi_mat.T @ t
    
    return Mn


def BLR(train_data, test_data_feature, O1=2, O2=3):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    weight = BLR_train(train_data, O1, O2)
    test_phi = np.transpose(phi(test_data_feature, O1, O2))
    y_BLRprediction = weight @ test_phi

    return y_BLRprediction 


def MLR(train_data, test_data_feature, O1=2, O2=3):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    weight = MLR_train(train_data, O1, O2)
    test_phi = np.transpose(phi(test_data_feature, O1, O2))
    y_MLLSprediction = weight @ test_phi

    return y_MLLSprediction 


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=2)
    parser.add_argument('-O2', '--O_2', type=int, default=3)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    
    #print('MSE of MLR= {e2}.'.format(e2=CalMSE(predict_MLR, data_test_label)))
    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))
    
    '''Plot part'''

    o1 = [x for x in range(2, 15)]
    o2 = [x for x in range(2, 15)]
    o1_list = []
    o2_list = []
    mlr = []
    blr = []
    
    for i in o1:
        for j in o2:
            o1_list.append(i)
            o2_list.append(j)
            mlr.append(CalMSE(MLR(data_train, data_test_feature, O1=i, O2=j), data_test_label))
            blr.append(CalMSE(BLR(data_train, data_test_feature, O1=i, O2=j), data_test_label))



    ax2 = plt.axes(projection='3d')
    ax2.scatter(o1_list, o2_list, mlr, c = 'r')
    ax2.set_title("MSE of MLR")
    plt.show()

    

    ax2 = plt.axes(projection='3d')
    ax2.scatter(o1_list, o2_list, blr, c = 'r')
    ax2.set_title("MSE of BLR")
    plt.show()


if __name__ == '__main__':
    main()