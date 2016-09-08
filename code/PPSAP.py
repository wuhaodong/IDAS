import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing as pre
from pprint import pprint
from math import ceil
from random import random, randint

#n_ads_list = [15,16,17,18,19,20,21,22,23,24,25] # number of ads (AD_model)
n_ads_list = [40,42,44,46,48,50,52,54,56,58,60]

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
    sam = sorted_AD_model[['ad_id','IR','LT','DIR','WPR']]
    #calculate the P of each ad
    #sam['P'] = float(sam['WPR'])
    ad_size = len(sam.index)
    sam.loc[:,'P'] = pd.Series(np.random.randn(ad_size), index=sam.index)
    sum = 0
    for i in range(ad_size):
        sum += sam['IR'][i]*1.0/(sam['LT'][i]*sam['WPR'][i])
    for i in range(ad_size):
        sam['P'][i] = sam['IR'][i]*1.0/(sam['LT'][i]*sam['WPR'][i]) / sum
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
        #apm['IR'][i] = sam['IR'][ad_size-i-1:ad_size-i]     
        #apm['LT'][i] = sam['LT'][ad_size-i-1:ad_size-i]
        
    
    return apm



def Schedule(ad_pair_model):
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
      
        pair_sche = Schedule(apm)
        
        ps = play_schedule(apm, sorted_SOP_model, pair_sche)
        
        imp = impression_log(SOP_model, ps, AD_model)
        
        eva  = evaluate(SOP_model, play_schedule, AD_model, imp)
        eva_list.append(eva)
        #print eva

    pprint(eva_list)
    print 'mdhd'


if __name__ == "__main__":
    main()
