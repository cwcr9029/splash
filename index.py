from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import re
import string

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index ():
    if request.form:
        if "tweet_input" in request.form:
            tweet_input = request.form["tweet_input"]
            r = re.findall("@[\w]*", tweet_input)
            for i in r:
                tweet_input = re.sub(i, "", tweet_input)
            tweet_input = tweet_input.replace("\n", "")
            exclude = set(string.punctuation)
            tweet_input = "".join(char for char in tweet_input if char not in exclude)

            # Load trained model
            # 0 = negative 4  = positive
            pipeline = joblib.load("./SPLASH/depression_model.pkl")
            tweet_input = [(tweet_input)]
            output = pipeline.predict(tweet_input)
            if output == 0:
                print("Positive result")
            else:
                print("Negative result")
            return render_template("index.html", tweet_input=tweet_input[0], output=output[0])
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
