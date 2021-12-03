# Project Title

This is the logistic regression model application for challenge 12!


## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
  The purpose of the analysis is I want to see if I can train/create a logistic regression model that will allow me to determine
  if a certain candidate will result in a loan that is low risk or has a higher risk of default. These candidates are represented by
  7 pieces of information that the model uses to come up with a zero or one meaning zero for low risk and one for high risk of default.

* Explain what financial information the data was on, and what you needed to predict.
  The 7 pieces of information are: loan size, interest rate, income of candidate, debt to income ratio, number of accounts they have open, total debt, and how many derogatory marks they have on their credit report.  From here I now need to come up with a result of this person will be solid and result in a low risk loan (marked as a '0'), or this client will be at high risk of default (marked as a '1').

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
  The variable I needed to predict was if when I encounter any candidate's 7 pieces of information, if that candidate will result in a zero or one.  With the data I started with and doing a value_counts() function call I saw I had 75036 zeroes (healthy loans) and 2500
  ones (high risk of default loans).  We can see there is a healthy imbalance with the majority of the data being the healty loans.
  75036/(75036+2500) = 0.9677 or 96.77% of the loans were healty loans.

* Describe the stages of the machine learning process you went through as part of this analysis.
  As we are taught in the training module we model, fit, and then predict.  We start by instantiating an instance of the 
  logistic regression. We do that by making a call to the LogisticRegression() function.  I then take the data and split them in a training set and a testing set. It is with the first training set that I make a call to the model.fit() function and allow the regression model to sort of practice. It is a scary thing! 

  After the fitting is complete I then run a predict() function call on the test set of data and I am able to compare how the model did
  on the data.  

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
  I do make a logisticRegression() function call on both the original data and the resampled data. I use predict() and fit() functions as previously mentioned.  For the resampled data I used the oversampling technique by calling the RandomOverSampler() function to make the zeroes and ones equal in number.  In this particular case the number of zeroes and ones were now equal at 56271. That's a big differences for the ones!

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

Just to define real quick what is Accuracy, precision, and recall I think would be helpful.

Accuracy = (number of True Positives + number of True Negatives) / Number of all datapoints (True and False Positives and Negatives)
Precision = (number of true positives) / (number of true positives + number of false positives)
So this number answers of all the positives you got, how many were correctly identified as positives?

Recall = (number of true positives) / (number of true positives + number of false negatives)
This metric asks of all the real world positives that were in the dataset, how many of them did the model identify?

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

  Accuracy = 0.952 or 95.2 %
  Precision of 1 = 0.85 or 85%
  Recall of 1 = .91 or 91%
  Precision of 0 = 1.0 or 100%
  Recall of 0 = .99 or 99%
So for this first model we had an excellent overall accuracy of 95%.  We also had a decent precision and when we predicted a loan
to be of high risk, we were correct that it was indeed a high risk loan 85% of the time.  Likewise for all the high risk loans
that were out there, we were able to identify 91% of them.

For the low risk loans, the model was nearly perfect. For all the loans that were predicted to be positive, they were all indeed positive, and of all the low risk loans that were out there, we identified and picked them up 99% of the time.


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  Accuracy = 0.9937 or 99.37% 
  Precision of 1 = .84 or 84%
  Recall of 1 = .99 or 99%
  Precision of 0 = 1.0 or 100%
  Recall of 0 = .99 or 99%

  This resampled model 2 is even better in my opinion. We have an overall accuracy of 99.37%.  The Precison is about the same for the high risk loans and we pick up 84% of them versus 85% previously.  The improvement is in the Recall metric. We now are able to pick up 99% of them where previously we only grabbed 91% of them.  You can see for the low risk '0' loans the precision and recall haven't
  changed with near perfection.



## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )  If you do not recommend any of the models, please justify your reasoning.

So how do we determine which model performs best in the end? I think it entirely depends on if you need to find the 1s or 0s more, or if there is nearly equal importance in finding both of them. In my opinion for this particular situation, I think because we know that the vast majority of loans in general are 0s, it is important to try and really polish the regression model to be able to spot the high risk loans as best as possible.  Finding the low risk loans or '0' loans isn't really going to be the problem.  The problem is being able to identify and weed out the high risk of default '1' loans. If you can do that, then you can maximize your business of giving out loans, and minimize any negative pnl issues that can result from high risk loans defaulting and blowing up in your face.

For the above reason I think the resampled model will be better for our situation. It has better accuracy and precision and recall in both 0 and 1. In our case the improvement in the recall in 1 loan identification is the biggest contributor to me choosing it over the original regression model.
---

## Technologies

I am using python version 3.7.10 and am importing the following from the built-in libraries and from functions i've created myself:

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')

---

## Installation Guide

I have python version 3.7.10 and git version 2.33.0.windows.2 installed on a laptop running windows 10 pro.

I launch the facebook colab webpage, upload forecasting_net_prophet.ipynb and that's it!


---

## Usage

Just upload the forecasting_net_prophet.ipynb notebook and run the code. User can feel free to change any dates used for charts or slicing if they want to study anything else.


---

## Contributors
Just me, Paul Lopez.


---

## License
No licenses required. Just install everything for free, pull from my repository, and enjoy!
