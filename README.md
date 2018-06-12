## Introductions

Collaborators: David Wolff and Nicolas Finkelstein

Emails: davidwolff2020@u.northwestern.edu and nicolasfinkelstein2020@u.northwestern.edu

Course: EECS 349 Machine Learning Northwestern University

### Task

The task of this project is to provide a prediction to players who are considering on surrendering early in a League of Legends game. Deciding on whether or not to surrender early not only saves time for the players, but also alleviates stress that may occur during the game.


### Approach

The data we used for this project was acquired [here](http://www.kaggle.com/paololol/league-of-legends-ranked-matches/data). This data contained a lot of features for each game and player. To decide whether or not the team should surreneder, the features we used had numeric values in order to organize our data. From this, we compared different learners and decided to use Decision Trees (CART) as our primary learner.

### Key Results

First, there were 6 learners that we were deciding on using. These learners are: Logistic Regression, Linear Discrimination Analysis, KN Neighbors Classifier, Decision Tree Classifier, Gaussian Naive Bayes, and Support Vector Machines. To decide which learner to use, only about 1000 samples were chosen from the entire data set to learn each type of learner. For each learner, we created learning curve graphs for both the validation set (which was 20% of the 1000 samples) and the training set. This can be seen below in the figures.

![Logistic Regression](/images/LR_figure.png)
![Linear Discriminant Analysis](/images/LDA_figure.png)
![K Neighbors Classifier](/images/KNN_figure.png)
![Decision Tree Classifier](/images/CART_figure.png)
![Gaussian Naive Bayes](/images/NB_figure.png)
![Support Vector Machines](/images/SVM_figure.png)

### Illustrations and Graphs

yo yo yo

### Final Report

hey hey hey
