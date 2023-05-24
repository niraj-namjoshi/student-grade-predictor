from flask import Flask, render_template,request,redirect,url_for
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
                  Medu = int(request.form['Medu'])
                  Fedu = int(request.form['Fedu'])
                  studytime = int(request.form['studytime'])
                  Failures = int(request.form['Failures'])
                  alchoholic = int(request.form['alchoholic'])
                  Dalc=0
                  Walc=0 
                  if(alchoholic==0):
                        Dalc=0
                        Walc=0
                  elif(alchoholic==1):
                        Dalc =0
                        Walc=1
                  elif(alchoholic==3):
                        Dalc=1
                        Walc=0
                  else:
                        Dalc=1
                        Walc=1
                                  
                                  
                  arraymaker = [Medu,Fedu,studytime,Failures,Dalc,Walc]
                  print(arraymaker)
                  colstransfer = ['Medu','Fedu','studytime','failures','Dalc','Walc']
                  userinp = pd.DataFrame(data=[arraymaker], columns=colstransfer)
                  w1 = modeltochkG1.predict(userinp)
                  w2 = modeltochkG2.predict(userinp)
                  w3 = modeltochkG3.predict(userinp)
                  n1=int(w1[0])
                  n2=int(w2[0])
                  n3=int(w3[0])
                  return redirect(url_for('result', n1=n1,n2=n2,n3=n3))
        
        return render_template('test_server.html')
    
@app.route('/result',methods = ['GET', 'POST'])
def result():
        n1 = request.args.get('n1')
        n2 = request.args.get('n2')
        n3 = request.args.get('n3')
        return render_template('submit.html', n1=n1,n2=n2,n3=n3)

    

if __name__ == '__main__':
    app.run(port=5100, debug = True)
    
