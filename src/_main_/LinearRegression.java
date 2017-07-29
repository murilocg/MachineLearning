package _main_;

import org.jblas.DoubleMatrix;

import control.CostFunction;
import control.GradientDescent;
import entity.util.Hypotesis;
import entity.util.LoadData;

public class LinearRegression {

	public static void main(String[] args) {

		System.out.println("Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/data_1.txt", ",");
		DoubleMatrix X = dataset.getColumn(0);
		DoubleMatrix Y = dataset.getColumn(1);
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, X.columns), X);
		DoubleMatrix theta = DoubleMatrix.zeros(2, 1);
		System.out.println("size training data: " + Y.length);
		System.out.println("\n\nRunning Cost Function before Gradient Descent...");
		double cost = CostFunction.compute(X, Y, theta);
		System.out.println("Cost found by cost function: " + cost);

		System.out.println("\n\nRunning Gradient Descent....");
		double alpha = 0.01;
		int iterations = 2000;
		System.out.println("alpha: " + alpha);
		System.out.println("iterations: " + iterations);
		theta = GradientDescent.compute(X, Y, theta, alpha, iterations);
		System.out.println("Theta found by gradient descent: " + theta.get(0) + ", " + theta.get(1));

		System.out.println("\n\nRunning Cost Function after Gradient Descent....");
		cost = CostFunction.compute(X, Y, theta);
		System.out.println("Cost found by cost function: " + cost);

		System.out.println("\n\nPredicting Prices....");
		double p1 = Hypotesis.compute(new DoubleMatrix(1, 2, new double[] { 1, 3.5 }), theta);
		System.out.println("For population = 35,000, we predict a profit of " + (p1 * 10000));
		double p2 = Hypotesis.compute(new DoubleMatrix(1, 2, new double[] { 1, 7 }), theta);
		System.out.println("For population = 70,000, we predict a profit of " + (p2 * 10000));
	}
}
