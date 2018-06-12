## Introductions

Collaborators: David Wolff and Nicolas Finkelstein

Emails: davidwolff2020@u.northwestern.edu and nicolasfinkelstein2020@u.northwestern.edu

Course: EECS 349 Machine Learning Northwestern University

### Task

The task of this project is to provide a prediction to players who are considering on surrendering early in a League of Legends game. League of Legends is a highly competitive, 5v5 online video game played by over 67 million people worldwide. Therefore, deciding on whether or not to surrender early not only saves time for the players, but also alleviates stress that may occur during the game since games can range anywhere from 30-50+ minutes.


### Approach

The data we used for this project was acquired [here](http://www.kaggle.com/paololol/league-of-legends-ranked-matches/data). The original dataset was divided up into 7 separate csv files consisting of both team-centered and individual player-centered data. Therefore, we cleaned the data and made edits to be able to merge the sets into one. Part of the process even involved creating a bag-of-words like attributes based on what champions were selected! To see the full data cleaning process, [click here](ADD_THE_HTML_FOR_CLEANING). Since we started off with approximately 240 attributes, we had to use our intuition based off our previous experiences with the game to narrow our list of attributes down to about 150 (138 being champions). This facilitated and sped up our training and testing. Once we had our data ready for model selection, we ran different learners and ultimately decided to use Decision Trees (CART) as our primary learner.

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
