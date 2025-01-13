import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle



def create_model(df):

    X = df.drop(['diagnosis'], axis =1)
    y = df['diagnosis']

    #scaled the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size= 0.2, random_state=42)

    #train
    model = LogisticRegression()
    model.fit(X_train,y_train)

    #test model
    y_pred = model.predict(X_test)
    print('The accuracy score is: ', accuracy_score(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))

    return model , scaler

def get_cleaned_data():

    df = pd.read_csv('Data/data.csv')
    df.drop(columns=['Unnamed: 32','id'], inplace=True)
    df['diagnosis'] = [1 if value =='M' else 0 for value in df.diagnosis]
    return df

def main ():

    df = get_cleaned_data()

    model, scaler = create_model(df)

    with open('Model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('Model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
if __name__ == '__main__' :
    main()
