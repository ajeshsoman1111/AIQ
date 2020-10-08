
import pickle
import numpy as np
import os
from flask import Flask,render_template,request
app = Flask(__name__)
loaded_model = pickle.load(open("randomrf.pkl", "rb")) 
@app.route('/')
@app.route('/home')
def home():
     return render_template('home.html')
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        in_feat=[int('0'+x) for x in request.form.values()]
        final_feat=np.array(in_feat).reshape(1, 6) 
        
        result = loaded_model.predict(final_feat) 
        return render_template("result.html", prediction =result)
if __name__ == '__main__':
    app.run(debug=True)