package _main_;

import org.jblas.DoubleMatrix;

import control.regression.LinearRegression;
import control.regression.LinearRegressionNormal;
import entity.model.Regression;
import entity.util.LoadData;

public class LinearRegressionMultiExample {

	public static void main(String[] args) {

		exampleLinearRegressionNormalize();
		
		exampleLinearRegressionNormalEquation();
	}
	
	private static void exampleLinearRegressionNormalize() {
		System.out.println("Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/data_2.txt", ",");
		DoubleMatrix X = dataset.getColumns(new int[] { 0, 1 });
		DoubleMatrix Y = dataset.getColumn(2);

		System.out.println("\n\nRunning Linear Regression Gradient Descent with Normalization....");
		Regression model = new LinearRegression(0.1, 400, true);
		model.train(X, Y);

		System.out.println("\n\nPredicting Prices....");
		DoubleMatrix predict1 = model.predict(new DoubleMatrix(1, 2, new double[] { 1650, 3 }));
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house: " + predict1.get(0));
	}
	
	private static void exampleLinearRegressionNormalEquation() {
		System.out.println("\n\nLoading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/data_2.txt", ",");
		DoubleMatrix X = dataset.getColumns(new int[] { 0, 1 });
		DoubleMatrix Y = dataset.getColumn(2);

		System.out.println("\n\nRunning Linear Regression Normal Equation....");
		Regression model = new LinearRegressionNormal();
		model.train(X, Y);

		System.out.println("\n\nPredicting Prices....");
		DoubleMatrix predict1 = model.predict(new DoubleMatrix(1, 2, new double[] { 1650, 3 }));
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house: " + predict1.get(0));
	}
}
