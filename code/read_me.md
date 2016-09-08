# Scheduling algorithms
data_gen.py (generate synthetic data for scheduling experiments)  
## one-by-one
RR.py  
D3.py  
DDIR.py  
DIW.py  

## AD-pair
RRAP.py  
PBAP.py  
PPSAP.py  
PPSAPA.py  
  
(These codes get the evaluation (#_of_failed_Ads and GM) with different number of Ads for different datasets)

# AD-model
WPR_predicting.py  
WPR_predicting_aggregated.py  
(Get the accuracy and MAE of NB, SVM, DT and LR)

# Demo
AFY.xml (grabbed from wordpress by Xuanyou)    
wp.py (convert AFY.xml to data.json, (filtered by creator david))  
schedule.sh (execute wp.py)    
data.json (input data for demo)  
DIW_demo.py (demo using DIW method)  
PPSAP_demo.py (demo using PPSAP method)   
output_DIW   
output_PPSAP  
