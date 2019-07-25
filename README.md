# Splash 2019: Deprevention
Sub-Theme: Healthcare/family and lifestyle

## Opportunity identified: Using AI to predict depression in children
Diagnosing children with depression is more problematic than diagnosing adults or teenagers, as some might not believe that depression in children is possible despite it being an actual clinical problem with different symptoms in children as compared to adults.

With AI, we can develop machine learning models based on children’s medical records, psychological questionnaires and voice analysis over a period of time, providing more accurate diagnosis. 

## Workflow

### Collecting useful and relevant data

#### Audio data
Initially, our team set out to obtain audio data that allows us to use Python packages such as PyAudio or SpeechRecognition, to process the data (by removing background noises as well as transcribing the audio and obtaining sentiment data). Examples of such data includes one where participants view an "animated emotionally evocative four-minute film ("The Present")", where "participants are prompted to narrate the story in their own words and answer a series of perspective-taking questions that are related to the content of the film" (http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/behavior.html#voice-facial-expressions). Other data online that allows us to collect such speech data also exists, but requires strict data protection and agreement requirement. As we have decided to open-source our project, all these limitations make us decide to go with open data, as explained below.

#### Twitter data
Using open dataset platform such as "Kaggle", we found an open Sentiment dataset (1.6 million datasets) (https://www.kaggle.com/kazanova/sentiment140), that have been extracted using the Twitter API.

### Processing data
Using data processing library Pandas, we first input the csv file that we have obtained. Following which, we check for null values ```tweets.isnull()``` to prevent anomalies that will affect the training of our model. Secondly, we remove unnecessary columns that we will not be using in order to reduce the data size (e.g. id, date, flag and user).

Next, we process the text data using the nltk (natural language text processing library), by removing patterns in the words as well as other characters (```@[\w]*``` etc.). Next, we remove stopwords, which are very typical language words used such as "I" or "ours".
