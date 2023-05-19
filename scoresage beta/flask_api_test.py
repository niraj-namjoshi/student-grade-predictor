from flask import Flask, render_template,request,redirect
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

app = Flask(__name__)

@app.route('/',methods = ['GET', 'POST'])
def hello_world():
        if request.method == 'POST':
            return redirect('/studentdet')
        
        return render_template('index.html')     

@app.route('/studentdet',methods = ['GET', 'POST'])
def studentdet():

    # machine learning code
        arraymaker=[0,0,0,0,0,0]
        w1 = [0.5]
        w2 = [0.6]
        w3 = [0.7]



        main_df = pd.read_csv(r"studentgrade.csv")

        #applying ordinal encoding to non numeric data

        from sklearn.preprocessing import OrdinalEncoder
        ordenc = OrdinalEncoder()
        ordenc_cols = ohc_cols = ['school','sex','address','famsize','Pstatus','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','Mjob','Fjob','reason']
        ordenc_X = pd.DataFrame(ordenc.fit_transform(main_df[ordenc_cols]))
        ordenc_X.columns = ordenc_cols
        main_df = main_df.drop(ordenc_cols, axis=1)
        main_df = pd.concat([main_df,ordenc_X],axis = 1)

        # applying stratified shuffle split to skewed data

        features = ['higher','studytime','Medu','Fedu','Walc','Dalc','school','failures','G1','G2','G3']
        traintochk = main_df[features]

        shuf =  StratifiedShuffleSplit(n_splits = 1,test_size=0.2,random_state=42)
        skewtochk = ['Medu','Fedu','studytime','failures','Dalc','Walc','G1','G2','G3']
        for trindx , tstindx in shuf.split(traintochk,traintochk[skewtochk].iloc[:,1]):
            df_traintochk = main_df.loc[trindx][skewtochk]
            df_testtochk = main_df.loc[tstindx][skewtochk]
        print(df_traintochk.head(1))

        # split to features and target value

        X_traintochk=df_traintochk.drop(['G1','G2','G3'],axis = 1)
        Y_traintochkG1=df_traintochk['G1']

        X_testtochk=df_testtochk.drop(['G1','G2','G3'],axis = 1)
        Y_testtochkG1=df_testtochk['G1']

        # model creation and training

        modeltochkG1 =  RandomForestRegressor(random_state = 42,max_leaf_nodes = 25)
        modeltochkG1.fit(X_traintochk,Y_traintochkG1)

        Y_traintochkG2=df_traintochk['G2']
        #Y_testtochkG2=df_testtochk['G2']
        Y_traintochkG3=df_traintochk['G3']
        #Y_testtochkG3=df_testtochk['G3']

        modeltochkG2 =  RandomForestRegressor(random_state = 42,max_leaf_nodes = 26)
        modeltochkG2.fit(X_traintochk,Y_traintochkG2)

        modeltochkG3 =  RandomForestRegressor(random_state = 42,max_leaf_nodes = 34)
        modeltochkG3.fit(X_traintochk,Y_traintochkG3)
        w1 = modeltochkG3.predict(X_testtochk.head(1))

        # ml code ends here
        if request.method == 'POST':
            Medu = request.form.get('Medu')
            Fedu = request.form.get('Fedu')
            studytime = request.form.get('studytime')
            Failures = request.form.get('Failures')
            Dalc= request.form.get("Dalc")
            Walc= request.form.get("Walc")
                                  
            arraymaker = [Medu,Fedu,studytime,Failures,Dalc,Walc]
            print(arraymaker)
            colstransfer = ['Medu','Fedu','studytime','failures','Dalc','Walc']
            userinp = pd.DataFrame(data=[arraymaker], columns=colstransfer)
            w1 = modeltochkG1.predict(userinp)
            w2 = modeltochkG2.predict(userinp)
            w3 = modeltochkG3.predict(userinp)
            return f'{w1}/20-{w2}/20-{w3}/20'
        numbers = [w1,w2,w3] 
        
        return render_template('main_page.html' , arraymaker = arraymaker)
    

if __name__ == '__main__':
    app.run(port=5000, debug = True)
    
