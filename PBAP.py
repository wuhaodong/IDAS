import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing as pre
from pprint import pprint
from math import ceil
from random import random

#n_ads_list = [15,16,17,18,19,20,21,22,23,24,25] # number of ads(input)
n_ads_list = [40,42,44,46,48,50,52,54,56,58,60]
n_play = 24*60*6/2 # number of ad-pair playing one day

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

def get_AD_model(n_ads):
    '''
        we use fake data (including ads to play(testing data))
    '''

    ads = pd.read_csv('data/ads_simulation_'+str(n_ads)+'.csv')
    return ads

    '''
    data = pd.read_csv('training_data.csv')

    # data preprocessing
    le = pre.LabelEncoder()
    data['category'] = le.fit_transform(data['category'])
    data['media'] = le.fit_transform(data['media'])
    data['color'] = le.fit_transform(data['color'])
    data['layout'] = le.fit_transform(data['layout'])
    data['originality'] = le.fit_transform(data['originality'])

    # features selection    
    fs_data = data[['ad_id','category','media','color','layout','originality','text_p']]
    
    # divide training and testing data
    X = fs_data[0:900]; T = fs_data[901:1000]
    y = data['WPR']
    y = y[0:900]

    # training
    clf = svm.SVC()
    clf.fit(X,y)

    # predicting
    result = clf.predict(T)
    T['WPR'] = result
    
    # drop duplicates
    T = T.drop_duplicates(['ad_id'])

    return T
    '''

def pair_ads(sorted_AD_model):
    sam = sorted_AD_model
    ad_size = len(sam.index)
    pair_size = int(ceil(ad_size/2))
    AD_pair_model = pd.DataFrame(np.random.randint(1, size=(pair_size,4)), columns=['ad0','ad1','IR','LT'])
    apm = AD_pair_model
    
    
    for i in range(pair_size):

        apm['ad0'][i] = sam['ad_id'][i:i+1]
        apm['ad1'][i] = sam['ad_id'][ad_size-i-1:ad_size-i]
        apm['IR'][i] = sam['IR'][ad_size-i-1:ad_size-i] # use the IR/LT of the weak ad    
        apm['LT'][i] = sam['LT'][ad_size-i-1:ad_size-i]
        
    return apm

def Calculate_Possibility(ad_info):
	# calculate the possibility matrix P
	ad = ad_info
	l = len(ad)
	P = [[0 for i in range(l)]for j in range(l)]
	for i in range(l):
		s = 0
		for k in range(l):
			if k == i: continue
			s += ad[k][1] * 1.0/(ad[k][2]**2) # tune the exponent
		for j in range(l):
			if i == j: continue
			P[i][j] = (ad[j][1]*1.0/(ad[j][2]**2))/s
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
	# Generate displaying schedule of one day
	ad = ad_info
	l = len(ad)
	sche = [] # schedule of one day
	sche.append(ad[0][0])
	if l == 1:
		sche = [ad[0][0] for i in range(n_play)]
		return sche
	for count in range(n_play-1): # display 4320 times per day
		former = sche[-1]
		for i in range(l):
			if ad[i][0] == former:
				r = random()
				for j in range(l):
					if I[i][j][0] <= r <= I[i][j][1]:
						sche.append(ad[j][0])
	return sche
	
def play_schedule(ad_pair_model, sorted_sop_model, pair_sche):
    ssm = sorted_sop_model
    apm = ad_pair_model
    play_schedule = ssm
    ps = play_schedule
    
    # add columns, ad0-ad5, to the frame
    ps['ad0'] = ps['label']; ps['ad1'] = ps['label']; ps['ad2'] = ps['label']
    ps['ad3'] = ps['label']; ps['ad4'] = ps['label']; ps['ad5'] = ps['label']
    
    # transform pair_sche to play schedule
    ps_length = len(ps.index)
    pair_length = len(pair_sche)
    for i in range(ps_length):
        for j in range(3):
            pair_id = pair_sche[3*i+j]
            ps['ad'+str(j*2)][i:i+1] = apm['ad0'][pair_id]
            ps['ad'+str(j*2+1)][i:i+1] = apm['ad1'][pair_id]

    # transform sorted schedule to real schedule
    ps = ps.sort_index()
    return ps

def impression_log(SOP_model, play_schedule, AD_model):
    impression_log = pd.DataFrame(np.random.randint(1, size=(24*60*6, 4)), columns=['count','AD','WPR','WPR+'])
    imp = impression_log
    imp_length = len(imp.index)
    imp.loc[:,'imp'] = pd.Series(np.random.randn(imp_length), index=imp.index)
    #return imp
    # assign count, AD
    ps = play_schedule
    ps_length = len(ps.index)
    for i in range(ps_length):
        for j in range(6):
            imp.loc[i*6+j,'count'] = ps['count'][i]/6.0
            imp.loc[i*6+j,'AD'] = ps['ad'+str(j)][i]

    # assign WPR, WPR+
    am = AD_model
    for i in range(imp_length):
        imp.loc[i,'WPR'] = am['WPR'][imp['AD'][i]]
        imp.loc[i,'WPR+'] = imp['WPR'][i]
    for i in range(1, imp_length, 2):
        if imp.loc[i,'WPR+'] >= 3: continue
        elif imp.loc[i-1,'WPR+'] == 5: imp.loc[i,'WPR+'] += 2       
        elif imp.loc[i-1,'WPR+'] == 4: imp.loc[i,'WPR+'] += 1

    # assign imp
    for i in range(imp_length):
        imp.loc[i,'imp'] = imp.loc[i,'count'] * imp.loc[i,'WPR+'] / 5.0

    imp = imp.sort_values(by='imp')
    return imp

def evaluate(SOP_model, play_schedule, AD_model, impression_log):
    # store impression of each ad
    di = AD_model[['ad_id','WPR','LT','DIR']]
    n_ads = len(di.index)
    di.loc[:,'DI'] = pd.Series([0]*n_ads, index=di.index) # initial
    imp = impression_log
    imp_filtered = imp[imp['imp']>0]
    if_length = len(imp_filtered.index)
    for i,row in enumerate(imp_filtered.values):
        count,ad,wpr,wpr2,each_imp = row
        di.loc[ad,'DI'] += each_imp
    
   # objective function    
    n_Pos = 0; n_Neg = 0
    Mul = 1; Sum = 0

    for i,row in enumerate(di.values):
        ad_id,WPR,LT,DIR,DI = row
        
        # Geometric mean
        Mul *= (abs(DI-DIR)+1)**np.sign(DI-DIR)
        if DI-DIR < 0: 
            n_Neg += 1
        
        # Arithmetic mean
        Sum += DI - DIR

    GM = Mul**(1./n_ads) -1
    AM = Sum/(1.*n_ads)

    return [n_ads, n_Neg, GM, AM]


def main():
    eva_list = []
    for i in range(len(n_ads_list)):
        n_ads = n_ads_list[i]

        SOP_model = get_SOP_model()
        #sorted_SOP_model = SOP_model
        sorted_SOP_model = SOP_model.sort_values(by=['label','count'],ascending=False)
        
        AD_model = get_AD_model(n_ads)
        sorted_AD_model = AD_model.sort_values(by='WPR',ascending=False)
        
        apm = pair_ads(sorted_AD_model)

        # transform apm to list to adapt DDIRx algorithm
        ad_pair_info = []
        pair_size = len(apm.index)
        for i in range(pair_size):
            ad = [i, apm['IR'][i], apm['LT'][i]]
            ad_pair_info.append(ad)
        P = Calculate_Possibility(ad_pair_info)
        I = Calculate_Interval(ad_pair_info, P)
        pair_sche = Schedule(ad_pair_info, I)

        ps = play_schedule(apm, sorted_SOP_model, pair_sche)

        imp = impression_log(SOP_model, ps, AD_model)

        eva = evaluate(SOP_model, play_schedule, AD_model, imp)
        eva_list.append(eva)
        #print eva

    pprint(eva_list)
    print "twdp"

if __name__ == "__main__":
    main()
