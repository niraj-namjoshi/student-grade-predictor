from flask import Flask, render_template,request,redirect
import pickle
import sklearn
import pandas as pd
import numpy as np
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
        loaded_model = pickle.load(open("grade1", "rb"))
        loaded_model2 = pickle.load(open("grade2", "rb"))
        loaded_model3 = pickle.load(open("grade3", "rb"))
        w1 = [0.5]
        w2 = [0.6]
        w3 = [0.7]
        # ml code ends here
        if request.method == 'POST':
            Medu = request.form.get('Medu')
            Fedu = request.form.get('Fedu')
            studytime = request.form.get('studytime')
            Failures = request.form.get('Failures')
            alcoholic = request.form.get("alchoholic")
            Dalc= request.form.get("Dalc")
            Walc= request.form.get("Walc")
                                  
            arraymaker = [Medu,Fedu,studytime,Failures,Dalc,Walc]
            print(arraymaker)
            colstransfer = ['Medu','Fedu','studytime','failures','Dalc','Walc']
            userinp = pd.DataFrame(data=[arraymaker], columns=colstransfer)
            w1 = loaded_model.predict(userinp)
            w2 = loaded_model2.predict(userinp)
            w3 = loaded_model3.predict(userinp)
            return f'{w1}/20-{w2}/20-{w3}/20'
        numbers = [w1,w2,w3] 
        
        return render_template('main_page.html' , arraymaker = arraymaker)
    

if __name__ == '__main__':
    app.run(port=3000, debug = True)
    