import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#Function to join soil types into 1 containing the existing soil type number
def get_soil_type(soil_types):
  for i in range(15, 55):
    if(soil_types[i]==1):
		return i-14
#Function to join wilderness areas into 1
def get_wilderness_area(wilderness_areas):
	for i in range(11, 15):
		if(wilderness_areas[i]==1):
			return i-10

if __name__ == "__main__":
  loc_train = "train.csv"
  loc_test = "test.csv"
  loc_submission = "kaggle.forest.submission.csv"
  loc_results="best.csv"
  df_train = pd.read_csv(loc_train)
  
  #Join soil types for training df
  df_train['Soil_Type']=df_train.apply(get_soil_type, axis=1)
  
  #Join wilderness areas for training df
  df_train['Wilderness_Area']=df_train.apply(get_wilderness_area, axis=1)


  df_results=pd.read_csv(loc_results)

  df_test = pd.read_csv(loc_test)

  #Join soil types for testing df
  df_test['Soil_Type']=df_test.apply(get_soil_type, axis=1)
  #Join wilderness areas for testing df
  df_test['Wilderness_Area']=df_test.apply(get_wilderness_area, axis=1)

  df_test.fillna(0, None,0, True)
  df_train.fillna(0, None,0, True)
  
  #Create a new calculated feature EVDtH
  df_train['EVDtH'] = df_train['Elevation']-\
  df_train['Vertical_Distance_To_Hydrology']
  df_test['EVDtH'] = df_test['Elevation']-\
  df_test['Vertical_Distance_To_Hydrology']

  #Create a new calculated feature DtoH distance to hydrology
  df_train['DtoH']=np.sqrt(df_train['Vertical_Distance_To_Hydrology']**2 + \
  df_train['Horizontal_Distance_To_Hydrology']**2)
  df_test['DtoH']=np.sqrt(df_test['Vertical_Distance_To_Hydrology']**2 + \
  df_test['Horizontal_Distance_To_Hydrology']**2)

  #Create a new calculated feature DtoR distance to roadways
  df_train['DtoR']=df_train['Elevation']-\
  df_train['Horizontal_Distance_To_Roadways']
  df_test['DtoR']=df_test['Elevation']-\
  df_test['Horizontal_Distance_To_Roadways']

  #Create a new calculated feature EHDtH
  df_train['EHDtH'] = df_train['Elevation']-\
  df_train['Horizontal_Distance_To_Hydrology']
  df_test['EHDtH'] = df_test['Elevation']-\
  df_test['Horizontal_Distance_To_Hydrology']
  
  #Distance to Firepoints
  df_train['DtoF']=df_train['Elevation']-\
  df_train['Horizontal_Distance_To_Fire_Points']
  df_test['DtoF']=df_test['Elevation']-\
  df_test['Horizontal_Distance_To_Fire_Points']

  #Feature columns list    
  feature_cols = ['Elevation','EVDtH', 'EHDtH', 'Soil_Type',\
  'Wilderness_Area','Horizontal_Distance_To_Roadways',\
  'Horizontal_Distance_To_Fire_Points','Hillshade_9am', 'DtoH', 'DtoR', 'DtoF',\
  'Horizontal_Distance_To_Hydrology', 'Hillshade_Noon','Hillshade_3pm']
  
  #Initialize the estimator to be used in AdaBoost
  EC = ExtraTreesClassifier(criterion='entropy', n_jobs=-1,\
  n_estimators = 500, max_features=None, min_samples_split=1)
  #Initialize AdaBoost
  clf=AdaBoostClassifier(n_estimators=900, learning_rate=1, base_estimator=EC)

  #Create X_train, and X_test containing only our feature cols
  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]

  #Create y which only contains the Cover Type
  y = df_train['Cover_Type']
  
  #Separate training data into 2 parts, a part to train and a part to test on
  X,X_,Y,Y_ = cross_validation.train_test_split(X_train, y,test_size=0.2)
  
  #Create test_ids list which we will have to write into our submission file
  test_ids = df_test['Id']
  #Train our classifier on the splitted part of the training df
  clf.fit(X, Y)
  #Try predicting on the test part of the training df
  pred_test=clf.predict(X_)
  #Print the accuracy score we got
  print accuracy_score(pred_test, Y_)
  
  #Retrain the classifier on the full training df
  clf.fit(X_train, y)
  #Check scores by cross validation
  scores = cross_validation.cross_val_score(clf, X_train, y)
  print scores.mean()

  #Predict on the real testing df
  pred=clf.predict(X_test)
  #In this line I used an older file I submitted to Kaggle and got the best score
  #just to see a bit how is the script running.
  print accuracy_score(pred, df_results['Cover_Type'])

  #Write results into the file to be submitted to kaggle.
  with open(loc_submission, "wb") as outfile:
    outfile.write("Id,Cover_Type\n")
    for i in range(len(test_ids)):
      outfile.write("{0},{1}\n".format(test_ids[i],pred[i]))
