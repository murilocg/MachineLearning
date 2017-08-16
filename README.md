# Machine-Learning
This library is made to simplify the use of regression on machine learning

# Technologies
 - jblas-1.2.4
 - java 8
 
# Installation
At the moment, there is only one way available to use the API.

# From the source code
You can build the project from the source in this repository, export as a JAR file and Add to the Build Path of your project.

# API Overview

```java
  // Here, we load the data necessary to train the model.
		DoubleMatrix dataset = LoadData.load("train_data/data_1.txt", ",");
  
  // split the data in two: features and expected answers for each set of features(row)
		DoubleMatrix X = dataset.getColumn(0);
		DoubleMatrix Y = dataset.getColumn(1);
		
  /*define the model to use. Obs: first parameter is the learning coeficient, second parameter is the number of iterations to      *train the model e finally the last parameter is a question "Do you want to normalize the data?".
   */
		Regression model = new LinearRegression(0.01, 2000, false);
		model.train(X, Y);
   

  // After training the model we predict some values, just to test the model
		DoubleMatrix predict1 = model.predict(new DoubleMatrix(new double[] {3.5}));
		DoubleMatrix predict2 = model.predict(new DoubleMatrix(new double[] {7}));
		
		System.out.println("For population = 35,000, we predict a profit of " + (predict1.get(0) * 10000));
		System.out.println("For population = 70,000, we predict a profit of " + (predict2.get(0) * 10000));
  ```
