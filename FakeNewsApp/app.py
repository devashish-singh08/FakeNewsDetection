from flask import Flask, render_template, request
import joblib
import re
import string

app = Flask(__name__)

lr = joblib.load("lr_model.pkl")
rf = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def label(n):
    return "Fake ❌" if n == 0 else "Not Fake ✅"

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        news = request.form['news']
        news = wordopt(news)
        vect = vectorizer.transform([news])

        lr_pred = lr.predict(vect)[0]
        rf_pred = rf.predict(vect)[0]

        final_pred = lr_pred

        results = {
            "Logistic Regression (Final Model)": label(lr_pred),
            "Random Forest": label(rf_pred),
            "Final Prediction": label(final_pred)
        }

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
