from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np

model  = None

if model == None:
    
    model=pickle.load(open('xgb_pipeline.pkl','rb'))
    
app = Flask(__name__)

@app.route('/',methods=['POST'])
def calculate():
    
    data = request.get_json()
    
    sex = float(dict(data)['SEX'])
    designation = float(dict(data)['DESIGNATION'])
    age = float(dict(data)['AGE'])
    unit = float(dict(data)['UNIT'])
    leaves_used = float(dict(data)['LEAVES USED'])
    leaves_remain = float(dict(data)['LEAVES REMAINING'])
    rating = float(dict(data)['RATING'])
    past_exp = float(dict(data)['PAST EXP'])
    
    output = model.predict([[sex,designation,age,unit,leaves_used,leaves_remain,rating,past_exp]])
    output = np.round(output[0],2)
    
    
    return jsonify(output)



if __name__=='__main__':
    app.run(debug=True)