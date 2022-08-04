from flask import Flask, request, render_template
import numpy as np
import math
import pickle
import sklearn

app = Flask(__name__, static_folder='static')

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #get data from user
    features_dict=request.form.to_dict()

    #convert to list
    features_list=list(features_dict.values())

    # log transforming features
    log_features=[]
    for i in features_list:
        log_features.append(math.log(float(i)+1))

    features_np=np.array(log_features)
    features_reshaped=np.reshape(features_np,(1,3))
    
    # predicting sales amount
    exp_pred=model.predict(features_reshaped)
    pred=round(math.exp(exp_pred[0]),2)-1

    return render_template('index.html',predicted_sales_amount=pred)


if __name__ == "__main__":
    app.run(debug=False)