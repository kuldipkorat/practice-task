import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,f1_score,r2_score

data = pd.read_csv('gym_members_exercise_tracking.csv')

class gym:
    def __init__(self,data:pd.DataFrame,model):
        self.data = data
        self.model = model
        
    def preprocessing(self,x_drop_column:list,y_column:str,label_column:list):
        label = LabelEncoder()
        for col in label_column:
            self.data[col] = label.fit_transform(self.data[col])
            
        x = self.data.drop(columns=x_drop_column)
        y = self.data[y_column]
        
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        
        return X_train,X_test,Y_train,Y_test
        
    def model_train(self,X_train,Y_train):
        self.model.fit(X_train,Y_train)
        return self.model
        
    def evaluate_model(self,X_test,Y_test):
        Y_pred = self.model.predict(X_test)
        accuracy = r2_score(Y_test,Y_pred)
        
        print(f"Your Model Accuracy : {round(accuracy*100,2)}")

if __name__ == "__main__":
    
    gym1 = gym(data,LinearRegression())
    
    x_drop_column = ['Age','Max_BPM','Avg_BPM','Resting_BPM','Workout_Type','BMI']
    X_train,X_test,Y_train,Y_test = gym1.preprocessing(x_drop_column,'BMI',['Gender'])
    
    gym1.model_train(X_train,Y_train)
    
    gym1.evaluate_model(X_test,Y_test)
