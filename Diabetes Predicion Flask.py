import numpy as np
from flask import Flask,request,render_template
import pickle

flask_app = Flask(__name__, template_folder="PycharmProjects\PythonProject\templates\dp_html.html")
model = pickle.load(open('diabetes.pkl', 'rb'))

@flask_app.route('/')
def home():
    return render_template("dp_html.html")

@flask_app.route('/predict',methods=['POST'])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features=[np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("dp_html.html",prediction_text="The person is {}".format(prediction))
if __name__=='__main__':
    flask_app.run(debug=True)

