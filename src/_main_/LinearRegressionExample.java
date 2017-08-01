package _main_;

import org.jblas.DoubleMatrix;

import control.regression.LinearRegressionGradientDescent;
import entity.model.LinearRegressionModel;
import entity.util.LoadData;

public class LinearRegressionExample {

	public static void main(String[] args) {

		System.out.println("Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/data_1.txt", ",");
		DoubleMatrix X = dataset.getColumn(0);
		DoubleMatrix Y = dataset.getColumn(1);
		
		System.out.println("\n\nRunning Linear Regression Gradient Descent....");
		LinearRegressionModel model = new LinearRegressionGradientDescent(0.01, 2000, false);
		model.train(X, Y);

		System.out.println("\n\nPredicting Prices....");
		DoubleMatrix predict1 = model.predict(new DoubleMatrix(new double[] {3.5}));
		DoubleMatrix predict2 = model.predict(new DoubleMatrix(new double[] {7}));
		
		System.out.println("For population = 35,000, we predict a profit of " + (predict1.get(0) * 10000));
		System.out.println("For population = 70,000, we predict a profit of " + (predict2.get(0) * 10000));
	}
}
