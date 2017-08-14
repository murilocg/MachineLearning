package control.regression;

import org.jblas.DoubleMatrix;

import entity.dto.ConfigGradientDescent;
import entity.model.Regression;
import entity.util.BinaryClass;
import entity.util.gradient_descent.GradientDescent;
import entity.util.hypoythesis.HypothesisLogisticRegression;

public class LogisticRegressionMultiClass extends Regression {

	private int[] classes;
	private double lambda;

	public LogisticRegressionMultiClass(double alpha, int iterations, int[] classes, double lambda) {
		super(alpha, iterations, new HypothesisLogisticRegression());
		this.classes = classes;
		this.lambda = lambda;
	}

	@Override
	public void train(DoubleMatrix X, DoubleMatrix Y) {
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		theta = DoubleMatrix.zeros(classes.length, X.columns);
		for (int i = 0; i < classes.length; i++) {
			int klass = classes[i];
			System.out.println("		Training a model to the Class: " + klass);
			DoubleMatrix tk = GradientDescent.compute(new ConfigGradientDescent(X, BinaryClass.parse(Y, klass),
					theta.getRow(i).transpose(), hypothesis, alpha, lambda, iterations));
			theta.putRow(i, tk);
		}
	}

	@Override
	public DoubleMatrix predict(DoubleMatrix X) {
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		DoubleMatrix Y = new DoubleMatrix(X.rows, 2);
		for (int i = 0; i < X.rows; i++) {
			DoubleMatrix prediction = predictOneItem(X.getRow(i), Y.getRow(i));
			Y.putRow(i, prediction);
		}
		return Y;
	}

	private DoubleMatrix predictOneItem(DoubleMatrix rowX, DoubleMatrix rowY) {
		for (int j = 0; j < classes.length; j++) {
			double prediction = hypothesis.compute(rowX, theta.getRow(j).transpose()).get(0);
			if (prediction > rowY.get(1)) {
				rowY.put(0, classes[j]);
				rowY.put(1, prediction);
			}
		}
		return rowY;
	}
}
