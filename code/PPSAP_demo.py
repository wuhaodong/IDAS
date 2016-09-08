import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing as pre
from pprint import pprint
from math import ceil
from random import random, randint


def shuffle(ary):
    a=len(ary)
    b=a-1
    for d in range(b,0,-1):
        e=randint(0,d)
        if e==d:
            continue
        ary[d],ary[e]=ary[e],ary[d]
    return ary

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
    resudd = clf.predict(T)
    ads['WPR'] = resudd
    print ads
	
    return ads


def pair_ads(sorted_AD_model):
    sam = sorted_AD_model[['ad_id','ir','dd','WPR']]
    #calculate the P of each ad
    #sam['P'] = float(sam['WPR'])
    ad_size = len(sam.index)
    sam.loc[:,'P'] = pd.Series(np.random.randn(ad_size), index=sam.index)
    sum = 0
    for i in range(ad_size):
        sum += sam['ir'][i]*1.0/(sam['dd'][i]*sam['WPR'][i])
    for i in range(ad_size):
        sam['P'][i] = sam['ir'][i]*1.0/(sam['dd'][i]*sam['WPR'][i]) / sum
        sam['P'][i] = round(sam['P'][i],2)
    
    #slicing P
    df = pd.DataFrame(columns=['ad_id','P'])
    sam = sam[['ad_id','P']]
    
    for i,row in enumerate(sam.values):
        ad_id,P = row
        id = str(int(ad_id))
        
        rep = int(P*100)
        df2 = pd.DataFrame([[id, 0.01]]*rep, columns=['ad_id','P'])
        df = df.append(df2, ignore_index=True)
    
    ssam = df # sliced_sam

    ad_size2 = len(ssam.index)
    pair_size = int(ceil(ad_size2/2))
    AD_pair_model = pd.DataFrame(np.random.randint(1, size=(pair_size,2)), columns=['ad0','ad1'])
    apm = AD_pair_model
    
    
    for i in range(pair_size):
        apm['ad0'][i] = ssam['ad_id'][i:i+1]
        apm['ad1'][i] = ssam['ad_id'][ad_size2-i-1:ad_size2-i]
        #apm['ir'][i] = sam['ir'][ad_size-i-1:ad_size-i]     
        #apm['dd'][i] = sam['dd'][ad_size-i-1:ad_size-i]
        
    
    return apm



def Schedule(ad_pair_model):
    '''
            Generate the initial playing schedule of one day
            (this schedule is before using SOP-model!!!) 
    '''
    apm = ad_pair_model
    pair_size = len(apm.index)
    sche = range(0,pair_size)
    sche = shuffle(sche) #shuffle
   
    round_l = len(sche) # length of 1 round
    day_l = 24*60*6/2
    round_n = day_l / round_l + 1
    sche = sche * round_n
    sche = sche[:day_l]
    return sche

def play_schedule(AD_model, ad_pair_model, sorted_sop_model, pair_sche):
    '''
            This schedule is after using SOP-model
    ''' 
    ssm = sorted_sop_model
    apm = ad_pair_model
    play_schedule = ssm
    ps = play_schedule
    am = AD_model
	
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
    
	# sche_SOP_id (convert back to list format)
    sche_SOP_id = [] # just a list of ad_id
    for i in range(ps_length):
        for j in range(6):
            sche_SOP_id.append(ps['ad'+str(j)][i])

    # sche_SOP(add url...)
    sche_SOP = []
    for i in range(pair_length*2):
        ad_id = sche_SOP_id[i]
        sche_SOP.append([ad_id, am['WPR'][ad_id], am['url'][ad_id]]) 
    
    return sche_SOP




def main():
   

    SOP_model = get_SOP_model()
    sorted_SOP_model = SOP_model.sort_values(by=['label','count'],ascending=False)
       
    AD_model = get_AD_model()
    sorted_AD_model = AD_model.sort_values(by='WPR',ascending=False) 
    
    apm = pair_ads(sorted_AD_model)

    pair_sche = Schedule(apm)
    
    sche_SOP = play_schedule(AD_model, apm, sorted_SOP_model, pair_sche)

    f = open('output_PPSAP', 'w')
    for item in sche_SOP:
        #f.write(str(item[0])+' | ') # id
        #f.write(str(item[1])+' | ') # WPR
        f.write(item[2]+'\n')    


if __name__ == "__main__":
    main()
