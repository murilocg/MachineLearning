package _main_;

import org.jblas.DoubleMatrix;

import control.FeatureNormalize;
import control.GradientDescent;
import control.NormalEqn;
import entity.util.Hypotesis;
import entity.util.LoadData;

public class LinearRegressionMulti {

	public static void main(String[] args) {
		normalEquation();
		gradientDescent();
	}

	public static void normalEquation() {
		System.out.println("Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/data_2.txt", ",");
		DoubleMatrix X = dataset.getColumns(new int[] { 0, 1 });
		DoubleMatrix Y = dataset.getColumn(2);
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);

		System.out.println("\n\nRunning Normal Equation...");
		DoubleMatrix theta = NormalEqn.compute(X, Y);
		System.out.println("\n\nPredicting prices...");
		DoubleMatrix p = new DoubleMatrix(1, 3, new double[] { 1, 1650, 3 }).mmul(theta);
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house ...(using normal equation): " + p.get(0));
	}

	public static void gradientDescent() {

		System.out.println("Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/data_2.txt", ",");
		DoubleMatrix X = dataset.getColumns(new int[] { 0, 1 });
		DoubleMatrix Y = dataset.getColumn(2);

		System.out.println("\n\nNormalizing Features ...");
		FeatureNormalize X_feature = new FeatureNormalize(X);
		X = X_feature.normalize();
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);

		System.out.println("\n\nRunning Gradient Descent...");
		double alpha = 0.1;
		int iterations = 400;
		System.out.println("alpha: " + alpha);
		System.out.println("iterations: " + iterations);
		DoubleMatrix theta = DoubleMatrix.zeros(1, X.columns);
		theta = GradientDescent.compute(X, Y, theta, alpha, iterations);

		System.out.println("\n\nPredicting prices...");
		double sqtfeet = 1650;
		double br = 3;
		sqtfeet = X_feature.normalize(sqtfeet, 0);
		br = X_feature.normalize(br, 1);

		double price = Hypotesis.compute(new DoubleMatrix(new double[] { 1, sqtfeet, br }), theta);
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house ...(using gradient descent): " + price);
	}
}
