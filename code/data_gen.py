from random import randint, uniform
from random import choice
from random import random
from random import seed
from pprint import pprint
from math import ceil

# attributes
Categorie = ['sport','living','entertainment','painting']
Media = ['video']+['picture']*4
Color = ['red','blue','green']
Layout = ['skeleton','division','symmetry','full-page']
Originality = ['Y']+['N']*3
WPR_list = [4]+[3]*4+[2]*8+[1]*14

# generate fake data
play_data = []
ad_data = []
score_data = []
play_size = 1000
#ad_size_list = [15,16,17,18,19,20,21,22,23,24,25]
#ad_size_list = [20,21,22,23,24,25,26,27,28,29,30]
ad_size_list = [40,42,44,46,48,50,52,54,56,58,60]

for ad_size in ad_size_list:
   
    '''
    for i in range(play_size):
        id = i
        ad_id = randint(0,ad_size-1)
        ppl_c = randint(0,5)
        play_data.append([id,ad_id,ppl_c])
    '''
    for i in range(ad_size):
        # ad features
        ad_id = i
        
        '''
        category = choice(Categorie)
        media = choice(Media)
        color = choice(Color)
        text_p = round(random(),2)
        layout = choice(Layout)
        originality = choice(Originality)
        '''
        
        # generate WPR (real ads'WPR are not given, we have to predcit them)
        #WPR = uniform(0,0.6) # average WPR = 2
        #WPR = uniform(0,1) # average WPR = 3
        #WPR = uniform(0.4,1) # average WPR = 4
        #WPR = 0.5*MD + 0.2*OG + 0.3*TP
        #WPR = int(ceil(WPR*5))
        WPR = choice(WPR_list)

        # generate IR, LT and DIR
        IR = randint(200,400)
        #LT = IR/randint(10,20) # for strong/middle/weak
        #LT = IR/randint(2,10) # for simulation
        DIR = randint(WPR*2, WPR*3) # for sim2
        LT = IR/DIR
        '''
        # transform some features to score
        if media == 'video': MD = uniform(0.5,1)
        else: MD = uniform(0,0.5)
        if originality == 'Y': OG = uniform(0.5,1)
        else: OG = uniform(0,0.5)
        if text_p < 0.2: TP = 5 * text_p
        else: TP = -1.25*text_p + 1.25
        score_data.append([media,MD,originality,OG,text_p,TP])
        '''
        
        # 1:very low  2:low  3:middle  4:high  5:very high
        #print ad_size, ad_id
        ad_data.append([ad_id,IR,LT,DIR,WPR])
        #ad_data.append([ad_id,IR,LT,DIR,category,media,color,text_p,layout,originality,WPR])
        

    # write data into csv
    pd = play_data
    ad = ad_data
    f1 = open('data/ads_sim2_'+str(ad_size)+'.csv','w')

    f1.write('ad_id,IR,LT,DIR,WPR'+'\n')
    #print ad_size
    #pprint(ad)
    for i in range(ad_size):
        for j in range(5):
            if j == 4: f1.write(str(ad[i][j])+'\n')
            else: f1.write(str(ad[i][j])+',')
    f1.close()
    ad_data = []



'''
f1.write('ad_id,IR,LT,DIR,category,media,color,text_p,layout,originality,WPR'+'\n')
for i in range(ad_size):
    for j in range(11):
        if j == 10: f1.write(str(ad[i][j])+'\n')
        else: f1.write(str(ad[i][j])+',')
f1.close()
'''
#f2 = open('play_log.csv','w')
#f3 = open('IVA_log.csv','w')

'''
f2.write('id,ad_id,ppl_c'+'\n')
for i in range(play_size):
    for j in range(3):
        if j == 2: f2.write(str(pd[i][j])+'\n')
        else: f2.write(str(pd[i][j])+',')
f2.close()
'''
'''
f.write('id,ad_id,ppl_c,IR,LT,DIR,category,media,color,text_p,layout,originality,WPR'+'\n')
for i in range(play_size):
    for j in range(3):
        f.write(str(pd[i][j])+',')
    ad_id = pd[i][1]
    for k in range(1,11):
        if k == 10: f.write(str(ad[ad_id][k]))
        else: f.write(str(ad[ad_id][k])+',')
    f.write('\n')
f.close()
'''
