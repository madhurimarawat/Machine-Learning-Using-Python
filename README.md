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

<h2>2. Unsupervised Learning</h2>

--> Unsupervised learning is the training of a machine using information that is neither classified nor labeled and allowing the algorithm to act on that information without guidance.<br><br>
--> Here the task of the machine is to group unsorted information according to similarities, patterns, and differences without any prior training of data. <br>

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

---

# Libraries Used 📚 💻
<p>Short Description about all libraries used.</p>
<ul>
<li>NumPy (Numerical Python) – Enables with collection of mathematical functions
to operate on array and matrices. </li>
  <li>Pandas (Panel Data/ Python Data Analysis) - This library is mostly used for analyzing,
cleaning, exploring, and manipulating data.</li>
  <li>Matplotlib - It is a data visualization and graphical plotting library.</li>
<li>Scikit-learn - It is a machine learning library that enables tools for used for many other
machine learning algorithms such as classification, prediction, etc.</li>
  
</ul>

