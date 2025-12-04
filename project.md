<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Project Overview

In this project, I applied various ML techniques to demonstrate the Atmospheric and Oceanic Studies C111 course at UCLA, to study the impacts of both demographic factors and lifestyle choices on an individual's lifetime medical charges.


***

## Introduction 

Health care costs have risen dramatically over the past few decades, making it more important than ever for both policy issuers and insurance providers to understand the key contributors, as well as overall emerging trends. Specifically, the goal of this project is to provide a predictive framework that, when given demographic and lifestyle characteristics about an individual, can give a reasonable estimate of their lifetime insurance cost. In this project, I used a dataset publicly available from Kaggle, which considered the following attributes: age, sex, bmi, #of children, smoker, region when predicting the target variable charges( individual medical insurance costs in USD). This setup naturally caused me to pursue a supervised ml regression problem, where the target became the continuous variable charges. I sampled several regression approaches: Linear regression, Random Forest, Gradient Boosting, and even a Neural Network(MLP). These approaches were then evaluated on common ML metrics such as (R^2), mean absolute error(MAE), root mean squared error(RSME), and mean absolute percentage error(MAPE).

Through these experiments, my final conclusion was that the Random Forest approach performed the best, and offered the best predictive abilities for the problem statement out of any of the approaches. Furthermore, the key contributors were concluded as smoker status, BMI, and age, with smoking being the clear leading cause. 

## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/correlation_heatmap.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

<p>
When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

