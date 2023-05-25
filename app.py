import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("xgb_model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    #print(request.form.values())
    float_features = [float(x) for x in request.form.values()]
    vol_mov_avg = float_features[0]
    adj_close_rolling_median = float_features[1]
    
    input_data = pd.DataFrame({'vol_moving_avg': [vol_mov_avg], 'adj_close_rolling_med': [adj_close_rolling_median]})
    expected_features = input_data.columns.tolist()
    input_df = input_data[expected_features]
    input_array = input_df.values
    prediction = model.predict((input_array))
    return render_template("index.html", prediction_text = "The volume is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(host='0.0.0.0',port=8080)