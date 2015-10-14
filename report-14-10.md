### Dataset

We are using the Amazon product data available at http://jmcauley.ucsd.edu/data/amazon/ with appropriate permissions from the author.
Each review comes with users’ helpfulness votes and hence helpfulness score can be approximated using “X of Y approach.”
We consider the category **Cell Phone and Accessories** and construct a subset of reviews between 1995 and 2013 that have atleast 5 votes.
The total number of reviews thus considered is **147638**.

**Distribution of Helpflness scores:**

![Distribution](https://raw.githubusercontent.com/ankeshanand/masters-thesis/master/graphs/distribution%20of%20helpfulness.png "Distribution of Helpfulness scores")

### Features

**1. Structural:**
* Total number of tokens
* Total number of sentences
* Average length of sentences
* Number of exclamations.
* Percentage of Question sentences.

**2. Lexical:**
* Unigrams and Bigrams after removing stop-words, subsequent tf-idf weighting

**3. LIWC:**  A dictionary which helps users to determine the degree that any text uses positive or negative emotions, self-references
and other language dimensions.

**4. INQUIRER:**  A dictionary in which words are grouped in categories. It is basically a mapping tool which maps each word to some semantic tags, e.g., absurd
is mapped to tags NEG and VICE. 

**5. Features that represent informativeness:** 
* Features of a product that appear in the review.
* *TODO*

**6. Meta Data: (???)** 

*If we use this, we are moving away from representing helpfulness purely as a property of text.*
* Stars of a review

### Model

We model the problem of predicting review helpfulness score as a regression problem. We use SVMRegressor with linear kernel
provided by scikit-learn, which is based on LibSVM. Performance is evaluated by Root Mean Square Error (RMSE).
Ten-fold cross-validation is performed for all experiments.

| Feature       | RMSE Score |
| ------------- | ------------- |
| STR           | 0.2654        |
| UGR           | 0.2682        |
| LIWC          | 0.2488        |
| INQUIRER      | 0.2463        |
| Fusion-all    | 0.2394        |
