# Case Study 1 (10%)

## Build the following models to predict whether a given review is Positive (4 or 5 stars), Negative (2 or less stars) or Neutral on the Yelp Dataset

[Yelp Dataset](https://www.yelp.com/dataset/download)

1. Naive Bayes model using Count Vectors for Unigram

2. SVM model using TF-IDF Vectors with Unigram+Bigram

3. Naive Bayes model using TF-IDF Vectors for Unigram+Bigram

4. SVM model using Count Vectors for Unigram+Bigram

5. Naive Bayes model using One-Hot Vectors with Unigram+Bigram

---

- Make sure to tune your models extensively and explain your rationale for the tuning approaches used each iteration

- Do call out what else could be done to tune the model and how it would have helped (w/ some numbers) at the top/bottom of your notebook.

- Please make sure there are enough data points in the test set (>5000) for Confusion Matrix, AUC etc.

- Submit your python notebook run sequentially end to end as a .pdf file

- Add a table to your submission that details the below (This table only shows for 2 models but you need to add similarly for all 5 models)

---

| Task                                                      | Status                                                  | Results                                                                                                          | Individual Responsible |
| --------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------- |
| NB model with Unigram + Count Vectors                     | Config or the model trained                             | Train Time = ?                                                                                                   | `your-contribution`    |
| Training Data Check                                       | Confusion Matrix Built?                                 | F1 Score for positive = ? <br/> F1 Score for negative = ?                                                        | `your-contribution`    |
|                                                           | AUC Plotted?                                            | AUC = ?                                                                                                          | `your-contribution`    |
|                                                           | Accuracy Computed                                       | Accuracy = ?                                                                                                     | `your-contribution`    |
| Feature Engineering                                       | Categorical/Numerical Features Added?                   | Name 2 Features added?                                                                                           | `your-contribution`    |
| Cross Validation                                          |                                                         |                                                                                                                  |
| Interpretability                                          | Intrepretability Implemented? Local or Global ?         | 2 intresting Findings?                                                                                           | `your-contribution`    |
| Testing Data Check                                        | Confusion Matrix Built?                                 | F1 Score for Positive = ? <br/> F1 Score for Negative = ?                                                        | `your-contribution`    |
|                                                           | AUC plotted?                                            | AUC = ?                                                                                                          | `your-contribution`    |
|                                                           | Accuracy computed?                                      | Accuracy = ?                                                                                                     | `your-contribution`    |
| Next Steps                                                | List out 2-3 possible next steps for the this model     |
| Naive Bayes Model with One-Hot vectors and Unigram+Bigram | Type of Smoothing Used? Config of model trained?        | Train Time = ?                                                                                                   | `your-contribution`    |
| Training Data Check                                       | Confusion Matrix Built?                                 | F1 Score for Positive = ? <br/> F1 Score for Negative = ?                                                        | `your-contribution`    |
|                                                           | AUC plotted?                                            | AUC = ?                                                                                                          | `your-contribution`    |
|                                                           | Accuracy computed?                                      | Accuracy = ?                                                                                                     | `your-contribution`    |
| Feature Engineering                                       | Feature Weightages Added?                               | 2 Features with the Highest Weights?                                                                             | `your-contribution`    |
| Cross Validation                                          | Type of Cross Validation performed?                     | Findings of Cross Validation?                                                                                    | `your-contribution`    |
| Testing Data Check                                        | Confusion Matrix Built?                                 | F1 Score for Positive = ? <br/> F1 Score for Negative = ?                                                        |
|                                                           | AUC plotted?                                            | AUC = ?                                                                                                          |
|                                                           | Accuracy computed?                                      | Accuracy = ?                                                                                                     | `your-contribution`    |
| Next Steps                                                | List out 2-3 possible next steps for Naive Bayes Models |
| Data Preprocessing and Feature Engg                       | Negation Handled?                                       | # of negation patterns handled?                                                                                  | `your-contribution`    |
|                                                           | Class Separability Checked?                             | Featureset with best separability?                                                                               | `your-contribution`    |
|                                                           | Train and Test Handled Correctly?                       | Data Preprocessing and Feature Engg: Steps taken before train test split and steps taken after train test split? | `your-contribution`    |
