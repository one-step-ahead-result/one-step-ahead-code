# Library
import pandas as pd
import numpy as np
import datetime as dt
import time,datetime
import math
from math import sin, asin, cos, radians, fabs, sqrt
from geopy.distance import geodesic
from numpy import NaN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from IPython.display import Image
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import random 
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,roc_curve
import sys

EARTH_RADIUS=6371

# Common Utilities
def num2date(num):
	# Convert eventid in GTD to standard time format
	num = str(num)
	d = num[:4]+'/'+num[4:6]+'/'+num[6:8]
	tmp = dt.datetime.strptime(d, '%Y/%m/%d').date()
	return tmp
	
def num2date_(num):
	# Convert time of market data to standard time format
	num = str(num)
	d = num[:4]+'/'+num[5:7]+'/'+num[8:10]
	tmp = dt.datetime.strptime(d, '%Y/%m/%d').date()
	return tmp
	
def get_week_day(date):
	day = date.weekday()
	return day
 
def hav(theta):
	s = sin(theta / 2)
	return s * s
 
def get_distance_hav(lat0, lng0, lat1, lng1):
	# The distance between two points of a sphere is calculated by the haversine formula
	# Longitude and latitude convert to radians
	lat0 = radians(lat0)
	lat1 = radians(lat1)
	lng0 = radians(lng0)
	lng1 = radians(lng1)
 
	dlng = fabs(lng0 - lng1)
	dlat = fabs(lat0 - lat1)
	h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
	distance = 2 * EARTH_RADIUS * asin(sqrt(h))
 
	return distance	
	
# Load the population density data - https://sedac.ciesin.columbia.edu/data/set/spatialecon-gecon-v4

def load_eco(filename,country):
	basic_ec_file1 = filename
	basic_ec = pd.read_excel(basic_ec_file1, country,header=0)	# Load the page of Israel
	lonlat_list = []
	for i in range(basic_ec.shape[0]):
		temp = []
		temp.append(basic_ec.iloc[i]['LONGITUDE'])
		temp.append(basic_ec.iloc[i]['LAT'])
		lonlat_list.append(temp)
	return lonlat_list

# Make terrorist attack features

def gtd_one_hot(gtd):
	# Group the features at daily level
	gtd_grouped = gtd.groupby(gtd['Timestamp']).sum()
	# Occurrence measure
	gtd_grouped['occur_count'] = gtd.groupby(gtd['Timestamp']).size()
	# Maintain the max nightlight value each day
	gtd_grouped['nightlight'] = gtd.groupby(gtd['Timestamp'])['nightlight'].max()
	# Obtain the weekday of certain timestamp
	gtd_grouped['week'] = gtd.groupby(gtd['Timestamp'])['week'].mean()
	
	return gtd_grouped
	
def lag(df,col_name,count):
	# Shift the column
	for i in range(1,count+1):
		df[col_name + '_' + str(i)] = df[col_name].shift(i)
	return df
	
def compute_nl(lon,lat):
	# Map certain geographic position to corresponding value of nightlight intensity
	round_lon = round((lon+180)*37.5)
	round_lat = 6750-round((lat+90)*37.5)
	try:
		return nl[int(round_lat)][int(round_lon)]
	except:
		return 0
		
def contain_or_not(string,list_):
	if string in list_:
		return 1
	else:
		return 0	
		
def adjust_week(timestamp,week):
	# Adjust the weekend to friday
	if week == 5:
		return (timestamp+datetime.timedelta(days=2)).strftime("%Y/%m/%d")
	elif week == 6:
		return (timestamp+datetime.timedelta(days=1)).strftime("%Y/%m/%d")
	return timestamp.strftime("%Y/%m/%d")

# Make the market features

def get_market_data(start,end,ref,goal,host,user,password,db):
	
	con = pymysql.connect(host,user,password,db, charset='utf8' )
	
	# Reference Index
	cmd1 = "select * from " + ref + " where Timestamp >= " + start + ' and Timestamp <= ' + end 
	ref_df = pd.read_sql(cmd1, con)

	#Goal Index
	cmd2 = "select * from " + goal + " where Timestamp >= " + start + ' and Timestamp <= ' + end 
	goal_df = pd.read_sql(cmd2, con)
	
	return ref_df,goal_df
	
def get_diff(origin_md,name):
	
	md = origin_md.copy()
	
	str1 = 'logdiff_' + name
	str2 = 'twologdiff_' + name
	
	md['close_shift1'] = md['Trade Close'].shift(1)
	md['onediff'] = md['Trade Close'].diff()
	
	md['open_shift_minus1'] = md['Trade Open'].shift(-1)
	md['twodiff'] = md['open_shift_minus1']-md['close_shift1']
	
	md = md.dropna()
	
	md[str1] = md['onediff']/md['close_shift1'] < 0
	md[str2] = md['twodiff']/md['close_shift1'] < 0
	
	md_onediff = pd.DataFrame(md,columns = ['Timestamp',str1]).dropna()
	md_twodiff = pd.DataFrame(md,columns = ['Timestamp',str2]).dropna()
	
	return md_onediff,md_twodiff	
	
# Merge terrorist attack features and market features
def diff_merge(gtd_grouped,diff_list):
		
	for i in range(1,len(diff_list)):
		diff_feature = pd.merge(diff_list[i-1],diff_list[i],on='Timestamp')    
		
	diff_feature = diff_feature.dropna()
	
	diff_feature = pd.merge(gtd_grouped,diff_feature,on='Timestamp',how='right')
		
		
	return diff_feature
	
def lag_part(feature,lag_features,lag_numbers):
	for i in range(len(lag_features)):
		feature = lag(feature,lag_features[i],lag_numbers[i])
	return feature
	
def reset_df(df,target_col,index):
	cols = list(df)
	cols.insert(index, cols.pop(cols.index(target_col)))
	df = df.loc[:, cols]
	return df
	
def final_process(gtd_grouped,diff_list,lag_features,lag_numbers,target_col,future_drop_col):
	
	feature = diff_merge(gtd_grouped,diff_list)
	
	feature.sort_values("Timestamp",inplace=True)
	feature = feature.fillna(0)
	
	
	feature = lag_part(feature,lag_features,lag_numbers)
	
	
	feature = reset_df(feature,target_col,len(feature.columns.values)-1)
	feature = feature.drop(future_drop_col,axis=1)
	feature.rename(columns={target_col: 'target'}, inplace = True)
	feature = feature.dropna()
	
	return feature
	
def train_test(features,split_point):
	y = list(features['target'])
	X = features.drop(['target','Timestamp'],axis=1)
	x = X.values
	var_list = list(X)
	X_train,X_test,Y_train,Y_test = x[:split_point],x[split_point:],y[:split_point],y[split_point:]
	return X_train,X_test,Y_train,Y_test,var_list
	
def pr_(y_test,y_pred):
	realminus = 0
	predminus = 0
	correct = 0

	for ii in range(len(y_test)):
		if y_test[ii] == True:
			realminus += 1
		if y_pred[ii] == True:
			predminus += 1
		if y_test[ii] == True and y_pred[ii] == True:
			correct += 1
	if predminus == 0:
		precision = 1
	else:
		precision = correct/predminus
	recall = correct/realminus
	if recall == 0:
		precision,recall = 1,0
	return correct,predminus,correct,realminus
	
def split_pos_neg(feature,y_pred):
	# Display the performance in days with terrorist attacks and days without terrorist attacks
	testset = feature[cut_point:].copy()
	testset = testset.reset_index()
	pred_content = pd.Series(y_pred)
	testset['pred'] = pred_content
	
	testset1 = testset[(testset['occur_count'] >= 1)]
	y_pred_ = list(testset1['pred'])
	y_test_ = list(testset1['target'])
	precision, recall = pr(y_test_,y_pred_)
	f1 = 2*(precision*recall)/(precision+recall)
	print(precision, ' ',recall,' ',f1)
	print(classification_report(y_test_,y_pred_))
	
	testset1 = testset[(testset['occur_count'] == 0)]
	y_pred_ = list(testset1['pred'])
	y_test_ = list(testset1['target'])
	precision, recall = pr(y_test_,y_pred_)
	f1 = 2*(precision*recall)/(precision+recall)
	print(precision, ' ',recall,' ',f1)
	print(classification_report(y_test_,y_pred_))
	
def best_para(x_train,x_val,y_train,y_val):
	mf1=0
	mins=0
	maxd=0
	for j in range(5,10):
		for i in range(15,32): 
			clf = tree.DecisionTreeClassifier(min_samples_leaf = i,max_depth = j) 
			clf.fit(x_train, y_train)
			y_pred = clf.predict(x_test)
			y_pred_pre = clf.predict_proba(x_val)
			precision, recall = pr(y_val,y_pred)
			f1 = 2*(precision*recall)/(precision+recall)
			if(f1>mf1):
				mf1=f1
				mins=i
				maxd=j
	return mf1,mins,maxd
	
def train_dt(feature,cut_point,samples_leaf=1,depth=100):
	x_train,x_test,y_train,y_test,var_list = train_test(feature,cut_point)
	y_pred = clf.predict(x_test)
	y_pred_pre = clf.predict_proba(x_test,min_samples_leaf = samples_leaf,max_depth = depth)
	print(classification_report(y_test,y_pred))
	im = clf.feature_importances_
	print(im)
	precision, recall = pr(y_test,y_pred)
	f1 = 2*(precision*recall)/(precision+recall)
	print(precision, ' ',recall,' ',f1)
	split_pos_neg(feature,y_pred)
	
	
def experment_full_sample(feature, cut_point):
	
	
	# Market Only Baseline - Exp-FS
	test = pd.DataFrame(feature, columns=['Timestamp','logdiff_sp500','logdiff_sp500_1','twologdiff_is100_1','twologdiff_is100_2','logdiff_sp500_2','target'])
	train_dt(test,cut_point)

	# Exp-FS
	X_train,x_val,Y_train,y_val,var_list = train_test(features[:cut_point],val_cut_point)
	_,mins,maxd = best_para(x_train,x_val,y_train,y_val)
	train_dt(feature,cut_point)


def experment_terr(feature, cut_point):
	# Exp-Terr
	feature_ = feature.copy()
	feature_ = feature_[(feature_['occur_count'] >= 1)]
	val_cut_point_terr = 320
	cut_point_terr = 415

	## Market Only Baseline - Exp-Terr
	test = pd.DataFrame(feature_, columns=['Timestamp','logdiff_sp500','logdiff_sp500_1','twologdiff_is100_1','twologdiff_is100_2','logdiff_sp500_2','target'])
	train_dt(test,cut_point_terr)

	# Exp-Terr
	X_train,x_val,Y_train,y_val,var_list = train_test(feature_[:cut_point_terr],val_cut_point_terr)
	_,mins,maxd = best_para(x_train,x_val,y_train,y_val)
	train_dt(feature_,cut_point_terr)
	
	
def one_step_ahead(feature):

	# One step ahead - Need to load the terrorist attack data extract from news since startdate 

	# Merge GTD data prior to that startdata and terrorist attack data extract from news since startdate
	gtd_news = pd.read_excel('reuters.xlsx','Israel')

	## rechange the load GTD data part
	gtd = gtd_original[gtd_original['country'] == 97]
	gtd = gtd[gtd['iday']!=0]
	gtd['Timestamp'] = gtd['eventid'].apply(num2date)
	gtd = gtd[['Timestamp','latitude','longitude','nkill','nwound','city','provstate']]
	gtd = gtd.dropna()
	startdate = '2007-01-01'
	gtd = gtd[gtd['Timestamp'] < dt.datetime.strptime(startdate, '%Y-%m-%d').date()]
	gtd_news['Timestamp'] = gtd_news['Timestamp'].apply(num2date_)
	gtd = pd.concat([gtd,gtd_news])


	feature_all = feature.copy()
	feature = feature[feature['occur_count'] != 0]
	startdate = '2007-01-01'
	feature_train = feature[feature['Timestamp'] < dt.datetime.strptime(startdate, '%Y-%m-%d').date()]
	feature_test = feature[feature['Timestamp'] >= dt.datetime.strptime(startdate, '%Y-%m-%d').date()]
	test_time = list(feature_test['Timestamp'])

	# Market-only baseline and full-feature version for one-step-ahead
	fall_count = 0
	fall_predict_true = 0

	fall_predict_count = 0
	fall_predict_count_true = 0
	for i in range(len(test_time)):
			train_set = pd.concat([feature_train[-feature_train.shape[0]+i:], feature_test[0:i]])
			test_set = feature_test[i:i+1]
			test_set = test_set.drop([], 1)
			
			
			# market-only version
			# x_train,x_test,y_train,y_test,var_list_market = train_test(train_set[['Timestamp','logdiff_sp500','logdiff_sp500_1','twologdiff_is100_1','twologdiff_is100_2','logdiff_sp500_2','target']],train_set.shape[0])
			# full-feature version
			x_train,x_test,y_train,y_test,var_list = train_test(train_set,train_set.shape[0])
			
			time = str((test_set['Timestamp'].values)[0])
			y = list(test_set['target'])
			# market-only version
			# X = test_set[['logdiff_sp500','logdiff_sp500_1','twologdiff_is100_1','twologdiff_is100_2','logdiff_sp500_2']]
			# full-feature version
			X = test_set.drop(['target','Timestamp'],axis=1)

			# market-only version
			# clf = tree.DecisionTreeClassifier() 
			# full-feature version
			clf = tree.DecisionTreeClassifier(min_samples_leaf = 26) 
			clf.fit(x_train, y_train)
			y_pred = clf.predict(X)
			if y == [1]:
				fall_count += 1
				if y_pred == [1]:
					fall_predict_true += 1
			if y_pred == [1]:
				fall_predict_count += 1
				if y == [1]:
					fall_predict_count_true += 1



	plusprecision = fall_predict_count_true/fall_predict_count
	plusrecall = fall_predict_true/fall_count
	f1 = 2*(plusprecision*plusrecall)/(plusprecision+plusrecall)
	print(plusprecision,' ',plusrecall,' ',f1)
	print(fall_predict_count_true,' ',fall_predict_count,' ',fall_predict_true,' ',fall_count)

def main(argv):

	# Load the population density data - https://sedac.ciesin.columbia.edu/data/set/spatialecon-gecon-v4	
	lonlat_list = load_eco('basic_eco.xls',"Israel")
		
	# Load the nightlight data - https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144897/BlackMarble_2016_3km_gray_geo.tif

	gray_file = open("nightlight.csv","rb")
	nl_tmp = np.loadtxt(gray_file,delimiter=',',skiprows=0)
	gray_file.close()
	nl = np.array(nl_tmp)	

	# Load the GTD data - https://www.start.umd.edu/gtd/
	gtd_original = pd.read_excel('gtd90_17.xlsx')	

	gtd = gtd_original[gtd_original['country'] == 97]
	gtd = gtd[gtd['iday']!=0]
	gtd['Timestamp'] = gtd['eventid'].apply(num2date)
	gtd = gtd[['Timestamp','latitude','longitude','nkill','nwound','city','provstate']]
	gtd = gtd.dropna()

	# capital/cultural center/religious center labels - From Wikipedia
	capital = ['Jerusalem','Nazareth','Haifa','Ramla','Tel Aviv','Beersheva']
	cultural_center = ['Tel Aviv']
	religious_center = ['Jerusalem']
			
	gtd['capital'] = gtd['city'].apply(contain_or_not,args=(capital,))
	gtd['cultural_center'] = gtd['city'].apply(contain_or_not,args=(cultural_center,))
	gtd['religious_center'] = gtd['city'].apply(contain_or_not,args=(religious_center,))

	# One-hot encoding of provstate
	gtd = gtd.join(pd.get_dummies(gtd.provstate))

	gtd['week'] = gtd['Timestamp'].apply(get_week_day)
	gtd['Timestamp'] =  gtd.apply(lambda x :adjust_week(x['Timestamp'],x['week']),axis=1)
	gtd['Timestamp'] = gtd['Timestamp'].apply(num2date_)
	gtd['week'] = gtd['Timestamp'].apply(get_week_day)
	gtd['nightlight'] = gtd.apply(lambda row: compute_nl(row['longitude'], row['latitude']), axis=1)
	basic_ec[['LAT']] = basic_ec[['LAT']].apply(pd.to_numeric)
	basic_ec[['LONGITUDE']] = basic_ec[['LONGITUDE']].apply(pd.to_numeric)
	gtd = gtd.reset_index(drop=True)

	add_feature = ['POPGPW_2005_40']
	gtd = pd.concat([gtd, pd.DataFrame(columns=add_feature)])

	for i in range(gtd.shape[0]):
		distance = []
		lon = gtd.iloc[i]['longitude']
		lat = gtd.iloc[i]['latitude']
		for j in range(basic_ec.shape[0]):
			distance.append(geodesic((lonlat_list[j][1],lonlat_list[j][0]), (lat,lon)))
		min_index = distance.index(min(distance)) 
		for j in range(len(add_feature)):
			# Calculate the population density
			gtd.loc[i,add_feature[j]] = float(basic_ec.iloc[min_index][add_feature[j]]/basic_ec.iloc[min_index]['AREA'])
	gtd[add_feature] = gtd[add_feature].apply(pd.to_numeric)
	keep_geo = gtd.groupby('Timestamp').first()
	gtd_grouped = gtd_one_hot(gtd)
	gtd_grouped = gtd_grouped.reset_index('Timestamp')
	gtd_grouped['longitude'] = pd.Series(list(keep_geo['longitude']))
	gtd_grouped['latitude'] = pd.Series(list(keep_geo['latitude']))

	# In order to take normal day into account from 1989/12/31 to 2018/01/01
	b = dt.datetime.strptime('1989/12/31', '%Y/%m/%d').date()
	ind = []
	vi = []
	for x in range(12000):
		b += pd.Timedelta(days = 1)
		if b == dt.datetime.strptime('2018/01/01', '%Y/%m/%d').date():
			break		
		if get_week_day(b) == 5 or get_week_day(b) == 6:
			continue		
		ind.append(b)
		vi.append(1)
	ts = pd.Series(vi, index = ind)
	dict_ts = {'Timestamp':ts.index}
	df_day = pd.DataFrame(dict_ts)

	gtd_grouped = pd.merge(gtd_grouped,df_day,how='right')
	gtd_grouped = gtd_grouped.sort_values(by=['Timestamp'])
	gtd_grouped = gtd_grouped.reset_index()
	gtd_grouped = gtd_grouped.drop(['index'],axis=1)
	gtd_grouped.fillna(0,inplace=True)
	gtd_grouped['week'] = gtd_grouped['Timestamp'].apply(get_week_day)
	gtd_grouped = gtd_grouped.drop(['week'],axis=1)

	tmp = list(gtd_grouped)[1:]
	for i in list(list(gtd_grouped)[1:]):
		if i == 'Timestamp':
			continue
		gtd_grouped = lag(gtd_grouped,i,2)
	gtd_grouped = gtd_grouped[(gtd_grouped['occur_count']!=0) | (gtd_grouped['occur_count_1']!=0) | (gtd_grouped['occur_count_2']!=0)]
	gtd_grouped = gtd_grouped.fillna(0)

	for i in tmp:
		sum_str = 'sum_' + i
		str1 = i+'_1'
		str2 = i+'_2'
		if i == 'latitude' or i == 'longitude':
			gtd_grouped[sum_str] = pd.Series(list(gtd_grouped[str1]))
			gtd_grouped = gtd_grouped.drop([str1,str2],axis=1)
			continue
		if i == 'nightlight':
			# For nightlight, take the max
			gtd_grouped[sum_str] = gtd_grouped.apply(lambda row: max(row[str1], row[str2]), axis=1)
			gtd_grouped = gtd_grouped.drop([str1,str2],axis=1)
			continue
		# For other, take the sum
		gtd_grouped[sum_str] = gtd_grouped[str1] + gtd_grouped[str2]
		gtd_grouped = gtd_grouped.drop([str1,str2],axis=1)
		
	gtd_grouped = gtd_grouped.drop(['latitude','longitude','sum_latitude','sum_longitude'],axis=1)

	# Load the market data - From Reuters client	
	sheetname='ISRAEL TA 100'
	Is100 = pd.read_excel('market_date_19.xlsx',sheetname)
	sheetname='SP500'
	Sp500 = pd.read_excel('market_date_19.xlsx',sheetname)	

	is100 = Is100.iloc[::-1]
	sp500 = Sp500.iloc[::-1]

	is100['Timestamp'] = is100['Timestamp'].apply(num2date_)
	sp500['Timestamp'] = sp500['Timestamp'].apply(num2date_)

	is100 = is100[is100['Timestamp'] < dt.datetime.strptime('2018-01-01', '%Y-%m-%d').date()]
	sp500 = sp500[sp500['Timestamp'] < dt.datetime.strptime('2018-01-01', '%Y-%m-%d').date()]

	sp500_onediff,sp500_twodiff = get_diff(sp500,'sp500')
	is100_onediff,is100_twodiff = get_diff(is100,'is100')
			
	is100_twodiff = lag(is100_twodiff,'twologdiff_is100',2)
	is100_twodiff = is100_twodiff.dropna()
	sp500_onediff = lag(sp500_onediff,'logdiff_sp500',2)
	sp500_onediff = sp500_onediff.dropna()
	
	diff_list = [sp500_onediff,is100_twodiff]
	lag_features = []
	lag_numbers = []
	target_col = 'twologdiff_is100'
	future_drop_col = []
	feature = final_process(gtd_grouped,diff_list,lag_features,lag_numbers,target_col,future_drop_col)
	feature = feature.drop(['sum_POPGPW_2005_40','sum_nightlight'],axis=1)		

	print('Total number of features:',feature.shape[0])

	# Data splitting
	train = feature[:3502]
	val = feature[3502:3802]
	train[train['occur_count'] != 0].shape[0]
	val_cut_point = 3502
	cut_point = 3802
	
	mode = sys.argv[1]
	
	if mode == 1:
		experment_full_sample(feature, cut_point)
	elif mode == 2:
		experment_terr(feature, cut_point)
	elif mode == 3:
		one_step_ahead(feature)
	


if __name__ == '__main__':
	main()
	