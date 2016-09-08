import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing as pre
from pprint import pprint
from math import ceil
from random import random


n_ads_list = [8]
n_play = 24*60*6 # number of ad-pair playing one day

def get_SOP_model():
    '''
        we use 20151201's data as the preticted data temporally
    '''
    SOP = pd.read_csv('data/20151201.txt', index_col='time')
    # new colomn: 'label'
    SOP['label'] = SOP['count']
    # label SOP model: 1:low  2:middle  3:high
    for i in range(1440):
        if SOP['count'][i]<2: SOP['label'][i] = 1
        elif SOP['count'][i]<4: SOP['label'][i] = 2
        else: SOP['label'][i] = 3
    return SOP

def get_AD_model():

    '''
    	Training and predicting WPR
    '''
	
    ## get data
    # new input ads
    ads = pd.read_json('data.json',orient='records')
    ads["ad_id"] = ads.index
    print ads
    # historical ads info
    train = pd.read_csv('data/AD-model training data_new.csv')
    # combine for encoding
    data = train.append(ads, ignore_index=True)

    ## data preprocessing
    le = pre.LabelEncoder()   
    data['cg'] = le.fit_transform(data['cg'])
    #data['sub-cg'] = le.fit_transform(data['sub-cg'])
    data['media'] = le.fit_transform(data['media'])
    data['tp'] = le.fit_transform(data['tp'])
    data['cs'] = le.fit_transform(data['cs'])
    
    ## features selection    
    fs_data = data[['cg','media','tp','cs']] # not considering sub-cg in demo
	
    ## divide training and testing data
    X = fs_data[:56]; T = fs_data[56:]
    y = train['WPR_d']   
	
    ## training
    #clf = GaussianNB()
    #clf = tree.DecisionTreeClassifier()
    clf = svm.SVC()
    clf.fit(X,y)
    
    ## predicting
    result = clf.predict(T)
    ads['WPR'] = result
    
    
    ## write the WPR file
    f2 = open('output_WPR', 'w')
    for i in range(len(ads.index)):
        f2.write(ads['url'][i]+'\n')
        f2.write(ads['cg'][i]+' ')
        f2.write(ads['media'][i]+' ')
        f2.write(str(ads['tp'][i])+' ')
        f2.write(ads['cs'][i]+'\n')
        f2.write(str(ads['WPR'][i])+'\n'+'\n')


    print ads
	
    return ads


def Calculate_Possibility(ad_info):
	# calculate the possibility matrix P
	ad = ad_info
	l = len(ad)
	P = [[0 for i in range(l)]for j in range(l)]
	for i in range(l):
		s = 0
		for k in range(l):
			if k == i: continue
			s += ad[k][1] * 1.0/(ad[k][2]*ad[k][3]) # tune the exponent
		for j in range(l):
			if i == j: continue
			P[i][j] = (ad[j][1]*1.0/(ad[j][2]*ad[j][3]))/s
	return P

def Calculate_Interval(ad_info, P):
	# calculate the interval matrix I
	l = len(ad_info)
	I = [[[0, 0] for i in range(l)]for j in range(l)]
	for i in range(l):
		for j in range(l):
			if j == 0: I[i][j] = [0, P[i][j]]
			I[i][j] = [I[i][j-1][1], I[i][j-1][1]+P[i][j]]
	return I

def Schedule(ad_info, I):
    '''
            Generate the initial playing schedule of one day
            (this schedule is before using SOP-model!!!) 
    '''
    
    ad = ad_info
    l = len(ad)
    sche = [] # schedule of one day
    sche.append([ad[0][4],ad[0][3],ad[0][5]]) # [ad_id, WPR, url]
    if l == 1:
        sche = [[ad[0][4],ad[0][3],ad[0][5]] for i in range(n_play)]
        return sche
    for count in range(n_play-1): # display 4320 times per day
        former = sche[-1]
        for i in range(l):
            if ad[i][4] == former[0]:
                r = random()
                for j in range(l):
                    if I[i][j][0] <= r <= I[i][j][1]:
                        sche.append([ad[j][4],ad[j][3],ad[j][5]])
    return sche
	
def play_schedule(AD_model, sorted_sop_model, sche):
    '''
            This schedule is after using SOP-model
    '''
    
    am = AD_model
    ps = sorted_sop_model # ps = play_schedule
    
    # add columns, ad0-ad5, to the frame (6 ads per minute)
    ps['ad0'] = ps['label']; ps['ad1'] = ps['label']; ps['ad2'] = ps['label']
    ps['ad3'] = ps['label']; ps['ad4'] = ps['label']; ps['ad5'] = ps['label']
    
    # transform pair_sche to play schedule
    ps_length = len(ps.index)
    ad_length = len(sche)
    for i in range(ps_length):
        for j in range(6):
            ad_id = sche[6*i+j][0]
            ps['ad'+str(j)][i:i+1] = am['ad_id'][ad_id]
            
    # transform sorted schedule to real schedule
    ps = ps.sort_index()

    # sche_SOP_id (convert back to list format)
    sche_SOP_id = [] # just a list of ad_id
    for i in range(ps_length):
        for j in range(6):
            sche_SOP_id.append(ps['ad'+str(j)][i])

    # sche_SOP(add url...)
    sche_SOP = []
    for i in range(ad_length):
        ad_id = sche_SOP_id[i]
        sche_SOP.append([ad_id, am['WPR'][ad_id], am['url'][ad_id]]) 
    
    return sche_SOP



def main():

    SOP_model = get_SOP_model()
    sorted_SOP_model = SOP_model.sort_values(by=['label','count'],ascending=False)
    
    AD_model = get_AD_model()
    #sorted_AD_model = AD_model.sort_values(by='WPR',ascending=False)
    am = AD_model
    ad_size = len(am.index)
    ad_info = []
    for i in range(ad_size):
            ad = [i, am['ir'][i], am['dd'][i], am['WPR'][i], am['ad_id'][i], am['url'][i]]
            ad_info.append(ad)
    print ad_info
    P = Calculate_Possibility(ad_info)
    I = Calculate_Interval(ad_info, P)
    sche = Schedule(ad_info, I)
    #print 'sche: ', sche
    sche_SOP = play_schedule(am, sorted_SOP_model, sche)
  
    f = open('output_DIW', 'w')
    for item in sche_SOP:
        #f.write(str(item[0])+' | ') # id
        #f.write(str(item[1])+' | ') # WPR
        f.write(item[2]+'\n')

		
		
if __name__ == "__main__":
    main()
