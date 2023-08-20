# Machine-Learning-Using-Python
This repository contains machine learning programs in the Python programming language.
<br><br>
<img src="https://i.morioh.com/52c215bc5f.png" height=400 width=700>

---

# About Python Programming
--> Python is a high-level, general-purpose, and very popular programming language.<br><br>
--> Python programming language (latest Python 3) is being used in web development, Machine Learning applications, along with all cutting-edge technology in Software Industry.
.<br><br>
--> Python is available across widely used platforms like Windows, Linux, and macOS.<br><br>
--> The biggest strength of Python is huge collection of standard library .<br>

---

# Machine learning 🤖 🛠🧠
<br>
<img src="https://www.analytixlabs.co.in/blog/wp-content/uploads/2018/10/Artboard-20-1.png" height=400 width=700>

--> Machine learning is a method of data analysis that automates analytical model building.<br><br>
--> It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.<br><br>
--> Machine Learning algorithm learns from experience E with respect to some type of task T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E.<br>

---

# Steps of Machine learning
<br>
<img src="https://github.com/madhurimarawat/Machine-Learning-Using-Python/assets/105432776/9e6a29ba-b1b2-4c54-b5c2-1f33f9389bdb" height=400 width=700>

---

# Types of Machine Learning
<br>
<img src="https://github.com/madhurimarawat/Machine-Learning-Using-Python/assets/105432776/5ebb4d39-1515-4d7a-a9c2-ab75242af166" height=400 width=700>
<br>
<h2>1. Supervised Learning</h2>
--> Basically supervised learning is when we teach or train the machine using data that is well-labelled. <br><br>
--> Which means some data is already tagged with the correct answer.<br><br>
--> After that, the machine is provided with a new set of examples(data) so that the supervised learning algorithm analyses the training data(set of training examples) and produces a correct outcome from labeled data.<br><br>

<h3>i) K-Nearest Neighbors (KNN) </h3>
<br>
--> K-Nearest Neighbours is one of the most basic yet essential classification algorithms in Machine Learning.<br><br>
--> It belongs to the supervised learning domain and finds intense application in pattern recognition, data mining, and intrusion detection..<br><br>
--> In this algorithm,we identify category based on neighbors.<br><br>

<h3>ii) Support Vector Machines (SVM) </h3>
<br>
--> The main idea behind SVMs is to find a hyperplane that maximally separates the different classes in the training data. <br><br>
--> This is done by finding the hyperplane that has the largest margin, which is defined as the distance between the hyperplane and the closest data points from each class. <br><br>
--> Once the hyperplane is determined, new data can be classified by determining on which side of the hyperplane it falls. <br><br>
--> SVMs are particularly useful when the data has many features, and/or when there is a clear margin of separation in the data.<br><br>

<h3>iii) Naive Bayes Classifiers</h3>
<br>
--> Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. <br><br>
--> It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other.<br><br>
--> The fundamental Naive Bayes assumption is that each feature makes an independent and equal contribution to the outcome.<br><br>

<h3>iv) Decision Tree</h3>
<br>
--> It builds a flowchart-like tree structure where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.<br><br>
--> It is constructed by recursively splitting the training data into subsets based on the values of the attributes until a stopping criterion is met, such as the maximum depth of the tree or the minimum number of samples required to split a node.<br><br>
--> The goal is to find the attribute that maximizes the information gain or the reduction in impurity after the split.<br><br>

<h3>v) Random Forest</h3>
<br>
--> It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.<br><br>
--> Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.<br><br>
--> The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.<br><br>

<h3>vi) Linear Regression</h3>
<br>
--> Regression: It predicts the continuous output variables based on the independent input variable. like the prediction of house prices based on different parameters like house age, distance from the main road, location, area, etc.<br><br>
--> It computes the linear relationship between a dependent variable and one or more independent features. <br><br>
--> The goal of the algorithm is to find the best linear equation that can predict the value of the dependent variable based on the independent variables.<br>
<h3>Types of Linear Regression</h3>
<h4>1. Univariate/Simple Linear regression</h4>
--> When the number of the independent feature, is 1 then it is known as Univariate Linear regression.<br>
<h4>2. Multivariate/Multiple Linear regression</h4>
--> In the case of more than one feature, it is known as multivariate linear regression.<br><br>

<h3>vii) Logistic Regression</h3>
<br>
--> Logistic regression is a supervised machine learning algorithm mainly used for classification tasks where the goal is to predict the probability that an instance of belonging to a given class or not. <br><br>
--> It is a kind of statistical algorithm, which analyze the relationship between a set of independent variables and the dependent binary variables. <br><br>
--> It is a powerful tool for decision-making.<br><br>
--> For example email spam or not. <br>
<h3>Types of Logistic Regression</h3>
<h4>1. Binomial Logistic regression</h4>
--> In binomial Logistic regression, there can be only two possible types of the dependent variables, such as 0 or 1, Pass or Fail, etc.<br>
<h4>2. Multinomial Logistic regression</h4>
--> In multinomial Logistic regression, there can be 3 or more possible unordered types of the dependent variable, such as “cat”, “dogs”, or “sheep”.<br>
<h4>3. Ordinal Logistic regression</h4>
--> In ordinal Logistic regression, there can be 3 or more possible ordered types of dependent variables, such as “low”, “Medium”, or “High”.<br><br>

<h2>2. Unsupervised Learning</h2>
--> Unsupervised learning is the training of a machine using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance.<br><br>
--> Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data. <br><br>
--> Unsupervised learning models are utilized for three main tasks— association, clustering and dimensionality reduction.<br><br>

<h3>i) Association Rules</h3>
--> An association rule is a rule-based method for finding relationships between variables in a given dataset.<br><br>
--> These methods are frequently used for market basket analysis, allowing companies to better understand relationships between different products.<br><br>
--> Understanding consumption habits of customers enables businesses to develop better cross-selling strategies and recommendation engines.<br><br>
--> Examples of this can be seen in Amazon’s “Customers Who Bought This Item Also Bought” or Spotify’s "Discover Weekly" playlist.<br>

<h3>Types of Association Rules</h3>

<h4>1. Apriori Algorithm</h4>
--> Apriori is an algorithm for frequent item set mining and association rule learning over relational databases.<br><br>
--> It proceeds by identifying the frequent individual items in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the database.<br><br>
--> The frequent item sets determined by Apriori can be used to determine association rules which highlight general trends in the database.<br><br>
--> This has applications in domains such as market basket analysis.<br><br>

<h3>ii) Clustering</h3>
--> Clustering is a data mining technique which groups unlabeled data based on their similarities or differences.<br><br>
--> Clustering algorithms are used to process raw, unclassified data objects into groups represented by structures or patterns in the information.<br><br>
--> Clustering algorithms can be categorized into a few types, specifically exclusive, overlapping, hierarchical, and probabilistic.<br><br>

<h3>iii) Dimentionality Reduction</h3>
--> Dimensionality reduction is a technique used when the number of features, or dimensions, in a given dataset is too high.<br><br>
--> It reduces the number of data inputs to a manageable size while also preserving the integrity of the dataset as much as possible.<br><br>
--> It is commonly used in the preprocessing data stage.<br>
<h3>Types of Dimentionality Reduction</h3>
<h4>1. Principal component analysis</h4>
--> Principal component analysis (PCA) is a type of dimensionality reduction algorithm which is used to reduce redundancies and to compress datasets through feature extraction.<br><br>
--> This method uses a linear transformation to create a new data representation, yielding a set of "principal components."<br><br>
--> The first principal component is the direction which maximizes the variance of the dataset.<br><br>
--> While the second principal component also finds the maximum variance in the data, it is completely uncorrelated to the first principal component, yielding a direction that is perpendicular, or orthogonal, to the first component.<br>

---
# Dataset Used

<h2>Iris Dataset</h2>
--> Iris Dataset is a part of sklearn library.<br><br>
--> Sklearn comes loaded with datasets to practice machine learning techniques and iris is one of them. <br><br>
--> Iris has 4 numerical features and a tri class target variable.<br><br>
--> This dataset can be used for classification as well as clustering.<br><br>
--> In this dataset, there are 4 features sepal length, sepal width, petal length and petal width and the target variable has 3 classes namely ‘setosa’, ‘versicolor’, and ‘virginica’.<br><br>
--> Objective for a multiclass classifier is to predict the target class given the values for the four features.<br><br.
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>Breast Cancer Dataset</h2>
--> The breast cancer dataset is a classification dataset that contains 569 samples of malignant and benign tumor cells. <br><br>
--> The samples are described by 30 features such as mean radius, texture, perimeter, area, smoothness, etc. <br><br>
--> The target variable has 2 classes namely ‘benign’ and ‘malignant’.<br><br>
--> Objective for a multiclass classifier is to predict the target class given the values for the features.<br><br>
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>Wine Dataset</h2>
--> The wine dataset is a classic and very easy multi-class classification dataset that is available in the sklearn library.<br><br>
--> It contains 178 samples of wine with 13 features and 3 classes.<br><br>
--> The goal is to predict the class of wine based on the features.<br><br>
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>Naive bayes classification data</h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/datasets/himanshunakrani/naive-bayes-classification-data"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Naive bayes classification data"> </a><br><br>
--> Contains diabetes data for classification.<br><br>
--> The dataset has 3 columns-glucose, blood pressure and diabetes and 995 entries.<br><br>
--> Column glucose and blood pressure data is to classify whether the patient has diabetes or not.<br><br>
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>Red wine Quality Dataset</h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Red wine Quality Dataset" alt="Red wine Quality Dataset"> </a><br><br>
--> Input variables (based on physicochemical tests):<br><br>
<table>
  <td>1. fixed acidity </td>     <td>2. volatile acidity</td>    <td>3. citric acid </td>  <td>4. residual sugar</td>  <td>5. chlorides</td>
<td>6 - free sulfur dioxide</td>  <td>7 - total sulfur dioxide</td>   <td>8 - density </td>     <td>9 - pH</td>             <td> 10 - sulphates</td>  <td>11 - alcohol</td> </table>

--> Output variable (based on sensory data):<br>
<table><td>12 - quality (score between 0 and 10)</td></table> 
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>Cars Evaluation Dataset</h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/datasets/elikplim/car-evaluation-data-set"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Cars Evaluation Dataset" alt="Cars Evaluation Dataset"> </a><br><br>
--> Contains information about cars with respect to features like Attribute Values:<br><br>
<table>
<td>1. buying v-high, high, med, low </td>
<td>2.maint v-high, high, med, low </td>
<td>3.doors 2, 3, 4, 5-more </td>
<td>4. persons 2, 4, more </td>
<td>5. lug_boot small, med, big</td>  
<td>6.safety low, med, high</td>  </table>
--> Target categories are:<br><br>
<table>
  <td>1. unacc 1210 (70.023 %)</td>
  <td>2. acc 384 (22.222 %)</td>
  <td>3. good 69 ( 3.993 %)</td>
  <td>4. v-good 65 ( 3.762 %)</td></table>
--> Contains Values in string format.<br><br>
--> Dataset is not cleaned, preprocessing is required.<br>

<h2>Census/Adult Dataset </h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python/input"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Census/Adult Dataset" alt="Census/Adult Dataset"> </a><br><br>
--> Contains dataset of population in various parameters like employment,marital status,gender,ethnicity etc.<br><br>
--> Model need to predict if income is greater than 50K or not.<br><br>
--> Contains Values in string format.<br><br>
--> Dataset is not cleaned, preprocessing is required.<br>

<h2>Salary Dataset</h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression
"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Salary Dataset" alt="Salary Dataset"> </a><br><br>
--> Contains Salary data for Regression.<br><br>
--> The dataset has 2 columns-Years of Experience and Salary and 30 entries.<br><br>
--> Column Years of Experience is used to find regression for Salary.<br><br>
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>USA Housing Dataset</h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/code/gantalaswetha/usa-housing-dataset-linear-regression/input
"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Housing Dataset" alt="Housing Dataset"> </a><br><br>
--> Contains Housing data for Regression.<br><br>
--> This dataset has multiple columns-Area Population, Address etc and Price(Output) and 5000 entries.<br><br>
--> Rest of the Columns are used to find regression for Price.<br><br>
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>Credit Card Fraud Dataset</h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/input
"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Salary Dataset" alt="Salary Dataset"> </a><br><br>
--> Contains Fraud data for Regression.<br><br>
--> The dataset has 31 columns.<br><br>
--> Dataset is already cleaned,no preprocessing required.<br>

<h2>Market Bucket Optimization Dataset</h2>
--> Dataset is taken from: <a href="https://www.kaggle.com/datasets/dragonheir/basket-optimisation
"><img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" height =40 width=40 title="Salary Dataset" alt="Salary Dataset"> </a><br><br>
--> Contains various product data for Apriori or association algorithm.<br><br>
--> The dataset has 20 columns of data about various products.<br><br>
--> Dataset is already cleaned,no preprocessing required.<br>

---
### Deep Learning 🤖🛠🧠🕸️
--> Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers.<br><br>
--> These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing it to “learn” from large amounts of data.<br><br>
--> While a neural network with a single layer can still make approximate predictions, additional hidden layers can help to optimize and refine for accuracy.<br><br>
--> Deep learning drives many artificial intelligence (AI) applications and services that improve automation, performing analytical and physical tasks without human intervention.<br><br>
--> Deep learning technology lies behind everyday products and services (such as digital assistants, voice-enabled TV remotes, and credit card fraud detection) as well as emerging technologies (such as self-driving cars).<br>

---
# Libraries Used 📚 💻
<p>Short Description about all libraries used.</p>
To install python library this command is used- pip install library_name <br><br>
<ul>
<li>NumPy (Numerical Python) – Enables with collection of mathematical functions
to operate on array and matrices. </li>
  <li>Pandas (Panel Data/ Python Data Analysis) - This library is mostly used for analyzing,
cleaning, exploring, and manipulating data.</li>
  <li>Matplotlib - It is a data visualization and graphical plotting library.</li>
<li>Scikit-learn - It is a machine learning library that enables tools for used for many other
machine learning algorithms such as classification, prediction, etc.</li>
  <li>Mlxtend (machine learning extensions)- It is a library of extension and helper modules for Python's data analysis and machine learning libraries.</li>
</ul>

