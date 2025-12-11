from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("xgboost_productivity_model.pkl", "rb"))

@app.route("/")
def about():
    return render_template("home.html")

@app.route("/about")
def home():
    return render_template("about.html")

@app.route("/prediction")
def home1():
    return render_template("prediction.html")

@app.route("/submit")
def home2():
    return render_template("submit.html")

# Correct POST method
@app.route("/pred", methods=['POST'])
def predict():

    # Get form values
    quarter = request.form['quarter']
    department = request.form['department']
    team = request.form['team']
    targeted_productivity = request.form['targeted_productivity']
    smv = request.form['smv']
    wip = request.form['wip']
    over_time = request.form['over_time']
    incentive = request.form['incentive']
    idle_time = request.form['idle_time']
    idle_men = request.form['idle_men']
    no_of_style_change = request.form['no_of_style_change']
    no_of_workers = request.form['no_of_workers']
    year = request.form['year']
    month = request.form['month']
    day_num = request.form['day_num']
    weekday = request.form['weekday']
    
    
    

    # Correct feature order (must match training dataset)
    input_features = np.array([[ 
        int(quarter),
        int(department),
        int(team),
        float(targeted_productivity),
        float(smv),
        float(wip),        
        float(over_time),
        float(incentive),
        float(idle_time),
        int(idle_men),
        int(no_of_style_change),
        int(no_of_workers),
        int(year),
        int(month),
        int(day_num),
        int(weekday)        
    ]])

    # Predict
    prediction = model.predict(input_features)
    pred = prediction[0]   # extract float value

    # Text output
    if pred <= 0.3:
        text = "The employee data shows Average Productivity"
    elif pred <= 0.8:
        text = "The employee data shows Medium Productivity"
    else:
        text = "The employee is Highly Productive"

    return render_template("submit.html", prediction_text=text)

if __name__ == "__main__":
    app.run(debug=True)
