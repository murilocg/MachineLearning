package _main_;

import org.jblas.DoubleMatrix;

import control.FeatureNormalize;
import control.GradientDescent;
import control.LinearRegressionGradientDescent;
import control.LinearRegressionNormal;
import control.NormalEqn;
import entity.model.LinearRegressionModel;
import entity.util.Hypotesis;
import entity.util.LoadData;

public class LinearRegressionMulti {

	public static void main(String[] args) {

		System.out.println("Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/data_2.txt", ",");
		DoubleMatrix X = dataset.getColumns(new int[] { 0, 1 });
		DoubleMatrix Y = dataset.getColumn(2);

		System.out.println("\n\nRunning Linear Regression Gradient Descent with Normalization....");
		LinearRegressionModel model = new LinearRegressionGradientDescent(0.1, 400, true);
		model.train(X, Y);

		System.out.println("\n\nPredicting Prices....");
		DoubleMatrix predict1 = model.predict(new DoubleMatrix(1, 2, new double[] { 1650, 3 }));
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house: " + predict1.get(0));

		System.out.println("\n\nLoading Data...");
		dataset = LoadData.load("train_data/data_2.txt", ",");
		X = dataset.getColumns(new int[] { 0, 1 });
		Y = dataset.getColumn(2);

		System.out.println("\n\nRunning Linear Regression Normal Equation....");
		model = new LinearRegressionNormal();
		model.train(X, Y);

		System.out.println("\n\nPredicting Prices....");
		predict1 = model.predict(new DoubleMatrix(1, 2, new double[] { 1650, 3 }));
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house: " + predict1.get(0));
	}
}
