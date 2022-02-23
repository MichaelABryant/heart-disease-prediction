"""Flask deployment for UCI heart disease classifier."""

# Import libraries.
from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Create instance with import name of place the app is defined.
app = Flask(__name__)

# Load pickles for scaler, encoder, and classifier.
scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
enc = pickle.load(open('onehotencoder.pkl', 'rb'))
lr_clf = pickle.load(open('logisticregression.pkl', 'rb'))

# Render homepage.
@app.route('/',methods=['GET'])
def Home():
    """Serve homepage template."""
    return render_template('index.html')

# Render prediction.
@app.route("/predict", methods=['POST'])
def predict():
    """Serve predict template."""
    
    if request.method == 'POST':
          
        # Retrieve user input.
        age = int(request.form['age'])
        sex = str(request.form['sex'])
        cp = str(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = str(request.form['fbs'])
        restecg = str(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = str(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = str(request.form['slope'])
        ca = int(request.form['ca'])
        thal = str(request.form['thal'])
        
        # Convert user input into the form of the original dataset.
        if sex == 'Male':
            sex = 1
        elif sex == 'Female':
            sex = 0
            
        if cp == 'typical':
            cp = 1
        elif cp == 'atypical':
            cp = 2
        elif cp == 'non-anginal':
            cp = 3
        elif cp == 'asymptomatic':
            cp = 4
            
        if fbs == 'True':
            fbs == 1
        elif fbs == 'False':
            fbs == 0

        if restecg == 'Normal':
            restecg = 0
        elif restecg == 'Abnormality':
            restecg = 1
        elif restecg == 'Hypertrophy':
            restecg = 2
    
        if exang == 'Yes':
            exang = 1
        elif exang == 'No':
            exang = 0
        
        if slope == 'Upsloping':
            slope = 0
        elif slope == 'Flat':
            slope = 1
        elif slope == 'Downsloping':
            slope = 2
    
        if thal == 'Normal':
            thal = 1
        elif thal == 'Fixed':
            thal = 2
        elif thal == 'Reversible':
            thal = 3
        
        # Create a list of user input.
        user_input = [age, trestbps, chol, thalach, oldpeak, sex, cp, fbs,
                      restecg, exang, slope, ca, thal]
        
        # Transform data into usable form by classifier.
        transformed_input = transform_input(user_input)
        
        # Create an instance of BytesIO to render plot on prediction page.
        img = BytesIO()
        
        # Calculate result and probability of heart disease.
        result = lr_clf.predict(transformed_input)
        probability = lr_clf.predict_proba(transformed_input)
        
        # Input for barplot.
        x = ["Heart disease", "No heart disease"]
        height = [(1-probability[0][0])*100, probability[0][0]*100]
    
        # Plot barplot for heart disease probability.
        bars = plt.bar(x=x, height=height, color = "#65aabb")
        plt.ylabel('Probability (%)')
        plt.ylim(0, 100)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x()+.325, yval + .5, "{}%".format(round(yval,2)))
        
        # Save figure and convert to url for HTML.
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('result.html', prediction = result, plot_url=plot_url)         
                               
    else:
        
        return render_template('index.html')


def transform_input(user_input):
    """Transform user input for LogisticRegression classifier."""
    
    # Define array with numerical feature names.
    numerical = [
        'age',
        'trestbps',
        'chol',
        'thalach',
        'oldpeak',
    ]

    # Define array with categorical feature names.
    categorical = [
        'sex',
        'cp',
        'fbs',
        'restecg',
        'exang',
        'slope',
        'ca',
        'thal'
    ]
    
    # Combine numerical and categorical feature names.
    columns = numerical + categorical
    
    # Initialize user input dataframe.
    df_input = pd.DataFrame()
    
    # Insert user inputs into dataframe.
    for idx, name in enumerate(columns):
        df_input.loc[0,name] = user_input[idx]
    
    # Scale numerical user input data with MinMaxScaler pickle.
    scaled_input = pd.DataFrame(scaler.transform(df_input[numerical]))
    
    # Encode categorical user input data with OneHotEncoder pickle.
    encoded_input = pd.DataFrame(enc.transform(df_input[categorical]))
    
    # Combine user input into one dataframe.
    transformed_input = pd.concat([scaled_input, encoded_input], axis=1)
    
    # Get encoded features names for categorical variables.
    encoded_categorical = enc.get_feature_names(categorical)
    
    # Set dataframe column names.
    transformed_input.columns = numerical + list(encoded_categorical)
    
    return transformed_input

if __name__=="__main__":
    app.run(debug=True)