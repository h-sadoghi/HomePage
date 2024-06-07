
# Introduction to Pattern Recognition
### Author : [Hadi Sadoghi Yazdi](https://scholar.google.com/citations?user=Z3XAKb0AAAAJ&hl=en)

$$
\begin{split}
var[X] &= \int p(x) \bigg[ f(x) - \mathbb{E}[f(x)] \bigg]^2 dx \\
&= \int p(x) \bigg[  f(x)^2 -2 \mathbb{E}[f(x)] f(x) + \mathbb{E}[f(x)]^2 \bigg]dx\\
&= \int (p(x)f(x)^2 - 2p(x)\mathbb{E}[f(x)] f(x) + p(x)\mathbb{E}[f(x)]^2) dx\\
&= \int p(x)f(x)^2 dx - 2\int p(x)\mathbb{E}[f(x)]f(x)dx + \int p(x)\mathbb{E}[f(x)]^2 dx\\
&= \mathbb{E}[f(x)^2] - 2\mathbb{E}[f(x)] \int p(x)f(x)dx + \mathbb{E}[f(x)]^2 \int p(x)dx \\
&= \mathbb{E}[f(x)^2] - 2\mathbb{E}[f(x)]^2 + \mathbb{E}[f(x)]^2 \\
&= \mathbb{E}[f(x)^2] - \mathbb{E}[f(x)]^2
\end{split}
$$

<p style="text-align: center;">
    <img src="cheetah.jpg" width="300" height="200">
</p>
<p style="text-align: center;">Dedicated to those who have devoted themselves to guiding humanitye</p>

## Preface
The book "Pattern Recognition" covers four main topics: clustering, classification, feature reduction, and classifier fusion, emphasizing their cost functions. It integrates both theoretical and practical aspects, offering ready-to-use codes for students and researchers. The book's unique approach categorizes methods based on cost functions, providing a comprehensive review of techniques from past to present. It includes detailed explanations and practical coding support, making it a valuable resource for understanding and applying pattern recognition concepts.
### Lesson 1: Introduction and Overview
 Nowadays, every individual is accompanied by a set of sensors that continuously collect various types of data. Additionally, people themselves contribute to the data pool by uploading photos, videos, and text. Smart watches and mobile phones, which many people carry, are equipped with an array of sensors including gyroscopes, accelerometers, three-axis magnetometers (x, y, z), and force sensors, which are common in most smart watches. The diagram below, along with an image of a smartwatch, illustrates the sensors and their structure within the watch. Some smartwatches even provide sensors for electrocardiogram and blood oxygen levels.

<p style="text-align: center;">
    <img src="SmartWatch.JPG" width="300" height="200">
</p>
<p style="text-align: center;">The internal structure of the smart watch</p>

 **_Code for daily step count data from a smart watch will be included here._**

```python 
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        df = pd.read_csv('dailyActivity_merged.csv')
        x = df.iloc[:, 3].values
        y = df.iloc[:, 4].values
        plt.plot(x,y,'.r')
```
Also activity recognition from other view can be see in this my paper. 
Activity Recognition In Smart Home Using Weighted Dempster-Shafer Theory
Hadi Sadoghi Yazdi and et.al. - Int. J. Smart Home, 2013


Data shared by individuals on social platforms like Flickr, 500px, Telegram, WhatsApp, Instagram, etc., can be valuable for analysis and decision-making in providing solutions for a better life. Nowadays, processing such data requires learning systems that are trainable. This book covers the necessary topics for such systems, typically focusing on fundamental principles and mathematics needed in pattern recognition learning discussions. The learning process involves issues such as data type, data application to the learning system, learning model, learning cost function, and system generalizability, which will be gradually introduced in the pattern recognition course over time. In this course, the focus is mainly on the following sections:
 
1. **Dataset** (Captured from sensors)
2. **Model** (Linear or non-linear)
3. **error+loss function+Cost function**
4. **Solution method** (Learning Rule)

# Dataset
## Sensor Datasets with Feature Extraction
Sensor datasets, recorded by various sensors detecting environmental changes, are crucial for real-time monitoring, decision-making, predictive analysis, and automation.

##### Types of Sensors
*Temperature Sensors*: Measure temperature.
*Pressure Sensors*: Detect pressure variations.
*Accelerometers and Gyroscopes*: Measure acceleration and orientation.
*Proximity and Light Sensors*: Detect object presence and light intensity.
*Sound Sensors*: Capture audio signals.
*Chemical Sensors*: Monitor environmental changes.
*GPS Sensors*: Provide location data.
**_Data is not only collected from temperature sensors, but also from other types of sensors that gather information such as text, video, audio, and various environmental parameters._**
<p style="text-align: center;">
    <img src="TextVideoImageSpeech.JPG" width="300" height="80">
</p>
<p style="text-align: center;">Data can exist in any form: text, audio, video, and images</p>

## Feature Extraction
Sensor datasets often contain diverse information collected from various types of sensors.Feature extraction transforms raw sensor data into representative features for analysis, improving data interpretation and prepare for machine learning algorithm.
_For example_, in the following figure, the activity signal introduced in the above section is converted into a feature vector including mean, variance, skewness, and other features.
<p style="text-align: center;">
    <img src="Activity_Feature.jpg" width="300" height="200">
</p>
<p style="text-align: center;">Activity signal converted into a feature vector (mean, variance, skewness, etc.)</p>

### Some Examples of Feature extraction
#### Feature Extraction from Text: 
Feature extraction from text involves converting text data into numerical representations that can be used for machine learning models. One common method is using the Term Frequency-Inverse Document Frequency (TF-IDF) approach.

Here’s a simple explanation and Python code to perform TF-IDF feature extraction:

#### Concepts:
*Document*: A piece of text. *Corpus*: A collection of documents. *Term Frequency (TF)*: The frequency of a term 𝑡 in a document 𝑑. *Inverse Document Frequency (IDF)*: Measures how important a term is in the entire corpus.
The TF-IDF value increases with the number of times a term appears in a document but is offset by the frequency of the term in the corpus, to adjust for the fact that some words are generally more common than others.

**Steps to Compute TF-IDF:**
1-Calculate the Term Frequency (TF) for each term in each document.
2-Calculate the Inverse Document Frequency (IDF) for each term.
3-Multiply the TF and IDF values to get the TF-IDF score for each term in each document.

##### TF-IDF Formula
TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a numerical statistic intended to reflect how important a word is to a document in a collection or corpus.

##### Term Frequency (TF)
The term frequency TF(t,d) is the frequency of term t  in document  d.


<!-- <!-- <!-- <!-- $$\[
\text{TF}(t,d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
\]$$ --> 

<div align="center"><img style="background: white;" src="dMXrkG3hO4.svg"></div>  


#### Inverse Document Frequency (IDF)
The inverse document frequency IDF(t,D)  measures how important a term is across the entire corpus D .

<!-- $$\[
\text{IDF}(t,D) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right)
\]$$ --> 

<div align="center"><img style="background: white;" src="Rh4ylSBoCs.svg"></div>

Where:
-  N is the total number of documents in the corpus. Denumerator is the number of documents where the term  t appears (i.e., the document frequency of the term).

#### TF-IDF Score
The TF-IDF score for a term  t  in a document d is the product of its TF and IDF values.

<!-- $$\[
\text{TF-IDF}(t,d,D) = \text{TF}(t,d) \times \text{IDF}(t,D)
\]$$ --> 

<div align="center"><img style="background: white;" src="naE5cCrJaN.svg"></div>

This formula adjusts the term frequency of a word by how rarely it appears in the entire corpus, emphasizing words that are more unique to specific documents.

#### Python code 
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "If you love God, follow God so that",
    "He may also love you and forgive your sins, ",
    "for He is Forgiving and Merciful.",
    "God has given Him another creation, meaning we have given Him a soul,",
    "and 'I have breathed into him of My spirit.'",
    "Those who have faith have a greater love for God.",
    "I have hastened towards You, my Lord, to seek Your pleasure.",
    "Although interpretation in words is clearer,",
     "love without words is brighter."
    ]

# Create the TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the documents to get the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense format and print it
tfidf_dense = tfidf_matrix.todense()
print("TF-IDF Matrix:\n", tfidf_dense)

# Print the feature names
print("\nFeature Names:\n", feature_names)
```
**_The point here is that different words like conjunctions and verbs with various tenses are considered as part of the words. such as 'and' 'has' 'have' 'forgive' 'forgiving'_**
### solve this problem
To address the issue of different forms of words (like conjunctions, verbs in various tenses, etc.) being treated as separate terms, we can use techniques such as lemmatization and removing stopwords. Lemmatization reduces words to their base or root form, and removing stopwords eliminates common words that are typically not useful for feature extraction.

Here’s a Python code example using nltk and sklearn to perform these preprocessing steps before applying TF-IDF:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Sample documents
documents = [
    "If you love God, follow God so that",
    "He may also love you and forgive your sins, ",
    "for He is Forgiving and Merciful.",
    "God has given Him another creation, meaning we have given Him a soul,",
    "and 'I have breathed into him of My spirit.'",
    "Those who have faith have a greater love for God.",
    "I have hastened towards You, my Lord, to seek Your pleasure.",
    "Although interpretation in words is clearer,",
     "love without words is brighter."
    ]
# Download NLTK resources (only need to run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Get the list of stopwords
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Lemmatize and remove stopwords
    processed_tokens = [
        lemmatizer.lemmatize(token.lower(), pos='v')  # Specify pos='v' for verbs
        for token in tokens if token.lower() not in stop_words and token.isalnum()
    ]
    
    # Join tokens back to string
    return ' '.join(processed_tokens)

# Preprocess the documents
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Create the TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed documents to get the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Get the feature names (terms)
feature_names = vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense format and print it
tfidf_dense = tfidf_matrix.todense()

# Print the feature names
print("\nFeature Names:\n", feature_names)
```
#### Feature Extraction from Image: 
**_Feature extraction_** from images involves transforming raw image data into a set of representative features that can be used for analysis or machine learning tasks. Here's an overview of the process:

1. **Preprocessing**
Before extracting features, it's often necessary to preprocess the images to standardize them and remove noise. Common preprocessing steps include resizing, cropping, normalization, and noise reduction.

2. **Feature Extraction Techniques**
There are various techniques for extracting features from images. Some popular methods include:

a. _Histogram of Oriented Gradients ([HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients))_
HOG computes the distribution of gradient orientations in localized portions of an image. It's commonly used for object detection and recognition tasks.

b. _Scale-Invariant Feature Transform ([SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform))_
SIFT detects and describes local features in an image that are invariant to scale, rotation, and illumination changes. It's widely used in image matching and object recognition.

c. Convolutional Neural Networks ([CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network))
CNNs are deep learning models that automatically learn hierarchical features from images. They consist of convolutional layers that extract features at different levels of abstraction.

3. Feature Representation
Once features are extracted, they need to be represented in a suitable format for analysis or machine learning algorithms. This could involve reshaping them into vectors or matrices.

4. Application
Extracted features can be used for various tasks such as image classification, object detection, image retrieval, and content-based image retrieval.

Python Libraries for Image Feature Extraction
Popular Python libraries for image feature extraction include OpenCV, scikit-image, and TensorFlow.

Here's a simple example using scikit-image to extract HOG features from an image: