import pandas as pd
import numpy as np
from sklearn import svm, tree, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing as pre
from pprint import pprint
from math import ceil
from random import random

n_ads = 24*60*6/2 # /2 for pair



def main():
    '''
        we use fake data (including ads to play(testing data))
    '''

    
    
    data = pd.read_csv('data/AD-model training data.csv')
    test = pd.read_csv('data/AD-model testing data.csv')
    #data = data.reindex(np.random.permutation(data.index))
    #print data

    ## data preprocessing
    le = pre.LabelEncoder()
    
    data['Category'] = le.fit_transform(data['Category'])
    data['Sub-category'] = le.fit_transform(data['Sub-category'])
    data['Media'] = le.fit_transform(data['Media'])
    data['Text-percentage'] = le.fit_transform(data['Text-percentage'])
    data['Color-scheme'] = le.fit_transform(data['Color-scheme'])
    '''
    data['category'] = le.fit_transform(data['category'])
    #data['Sub-category'] = le.fit_transform(data['Sub-category'])
    data['media'] = le.fit_transform(data['media'])
    #data['Text-percentage'] = le.fit_transform(data['Text-percentage'])
    data['color'] = le.fit_transform(data['color'])
    '''
    #print data
    
    ## features selection    
    #fs_data = data[['ad_id','category','media','color']]
    #print fs_data
    fs_data = data[['Category','Sub-category','Media','Text-percentage','Color-scheme']] 
    #fs_test = test[['Category','Sub-category','Media','Text-percentage','Color-scheme']]
    ## divide training and testing data
    X = fs_data; T = fs_data
    print T
    y = data['WPR_d']
    

    ## training
    # classification
    clf = GaussianNB()
    #clf = tree.DecisionTreeClassifier()
    #clf = svm.SVC()
    clf.fit(X,y)
    
    # regression
    #regr = linear_model.LogisticRegression(C=1e5)
    #regr = linear_model.LinearRegression()
    #regr.fit(X,y)

    ## predicting
    result = clf.predict(T)
    #result = regr.predict(T)
    # result discretization
    '''
    new_result = []
    for WPR in result:
        new_WPR = ceil(WPR*5)
        new_result.append(new_WPR)
    T['WPR_c'] = result
    T['WPR_d'] = new_result
    '''
    T['WPR_d'] = result
    print T
    # drop duplicates
    #T = T.drop_duplicates(['ad_id'])
    #data = data.drop_duplicates(['ad_id'])
    #print "data:", data[['ad_id','WPR']]
    #print "T:", T
    
    # evaluation
    
    '''
        classification
    '''
    
    total_test = 56
    correct_test = 0
    total_dist = 0
    for i in range(56):
        if T['WPR_d'][i] == test['WPR_d'][i]:
            correct_test += 1
        else:
            total_dist += abs(T['WPR_d'][i]-test['WPR_d'][i])
    accuracy = correct_test * 1. / total_test
    ave_dist = total_dist * 1. / total_test 
    print "accuracy: ", accuracy
    print "ave_dist: ", ave_dist
    
    '''
        regression
    '''
    
    '''
    for row in enumerate(T.values):
        ad_id, category, media, color, WPR = row
        if data[str(ad_id)]
    '''


if __name__ == "__main__":
    main()
