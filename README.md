## Introductions

Collaborators: David Wolff and Nicolas Finkelstein

Emails: davidwolff2020@u.northwestern.edu and nicolasfinkelstein2020@u.northwestern.edu

Course: EECS 349 Machine Learning Northwestern University

### Task

The task of this project is to provide a prediction to players who are considering on surrendering early in a League of Legends game. Deciding on whether or not to surrender early not only saves time for the players, but also alleviates stress that may occur during the game.


### Approach

The data we used for this project was acquired [here](http://www.kaggle.com/paololol/league-of-legends-ranked-matches/data). This data contained a lot of features for each game and player. To decide whether or not the team should surreneder, the features we used had numeric values in order to organize our data. From this, we compared different learners and decided to use Decision Trees (CART) as our primary learner.

### Key Results

First, there were 6 learners that we were deciding on using. These learners are: [Logistic Regression](https://machinelearningmastery.com/logistic-regression-for-machine-learning/), [Linear Discriminant Analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis), [K Neighbors Classifier](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), [Decision Tree Classifier](https://en.wikipedia.org/wiki/Decision_tree_learning), [Gaussian Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier), and [Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine). To decide which learner to use, only about 1000 samples were chosen from the entire data set to learn each type of learner. For each learner, we created learning curve graphs for both the validation set (which was 20% of the 1000 samples) and the training set. This can be seen below in the Illustrations and Graphs section of this paper. 

At first glance of all of these figures, the Logstic Regression (LR) and the Linear Disciminant Analysis (LDA) curves look way higher than any of the other figures. But we chose to use Decision Trees (CART) instead because we wanted to know which qualities of the beginning of the match will most likely determine the end result, win or lose. Considering that the accuracy for Decision Trees are also pretty accurate and nice, we chose to use CART. 

Once we chose CART as our primary learner, we finalized that the most important features/factors that occur in the game that decides victory are which team has gotten first tower, first dragon, and Rift Herald. Thse findings will be explained in the Final Report.


### Illustrations and Graphs

In the figures, the shaded area around the plots are the first standard deviations and score is based on accuracy, where highest accuracy is 1 (100%).

![Logistic Regression](/images/LR_figure.png)
![Linear Discriminant Analysis](/images/LDA_figure.png)
![K Neighbors Classifier](/images/KNN_figure.png)
![Decision Tree Classifier](/images/CART_figure.png)
![Gaussian Naive Bayes](/images/NB_figure.png)
![Support Vector Machines](/images/SVM_figure.png)


### Final Report

hey hey hey
