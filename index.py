from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from nltk.stem import PorterStemmer
import re
import string

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index ():
    if request.form:
        if "tweet_input" in request.form:
            tweet_input = request.form["tweet_input"]
            r = re.findall("@[\w]*", tweet_input)
            tweet_processed = tweet_input
            for i in r:
                tweet_processed = re.sub(i, "", tweet_processed)
            tweet_processed = tweet_processed.replace("\n", "")
            exclude = set(string.punctuation)
            tweet_processed = "".join(char for char in tweet_input if char not in exclude).lower()
            stemmer = PorterStemmer() 
            sw = ['i', 'me', 'my', 'myself', 'we', 'our', 
            'ours', 'ourselves', 'you', "youre", "youve", "youll", "youd", 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers', 'herself', 'it', 
             "its", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "thatll", 'these', 'those', 'am', 'is', 'are', 'was', 
              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 
              'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
              'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
              'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
              'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
              "dont", 'should', "shouldve", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "arent", 'couldn', 
              "couldnt", 'didn', "didnt", 'doesn', "doesnt", 'hadn', "hadnt", 'hasn', "hasnt", 'haven', "havent", 'isn', 
              "isnt", 'ma', 'mightn', "mightnt", 'mustn', "mustnt", 'needn', "neednt", 'shan', "shant", 'shouldn', 
              "shouldnt", 'wasn', "wasnt", 'weren', "werent", 'won', "wont", 'wouldn', "wouldnt"]
            Tweet = []
            for word in tweet_processed.split():
                if word not in sw:
                    Tweet.append(stemmer.stem(word))
            # Load trained model
            # 0 = negative 4  = positive
            pipeline = joblib.load("./SPLASH/depression_model.pkl")
            tweet_processed = [' '.join(Tweet)]
            output = pipeline.predict(tweet_processed)
            if output[0] == 0:
                result = "Negative Result"
            else:
                result = "Positive Result"
            return render_template("index.html", tweet_input=tweet_input, output=result,processed = tweet_processed[0])
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
