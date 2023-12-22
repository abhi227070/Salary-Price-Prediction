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
    
    sex = str(dict(data)['SEX'])
    designation = str(dict(data)['DESIGNATION'])
    age = int(dict(data)['AGE'])
    unit = str(dict(data)['UNIT'])
    leaves_used = int(dict(data)['LEAVES USED'])
    leaves_remain = int(dict(data)['LEAVES REMAINING'])
    rating = float(dict(data)['RATING'])
    past_exp = float(dict(data)['PAST EXP'])
    
    output = model.predict([[sex,designation,age,unit,leaves_used,leaves_remain,rating,past_exp]])
    output = float(np.round(output[0],2))
    
    
    return jsonify(output)



if __name__=='__main__':
    app.run(debug=True)
