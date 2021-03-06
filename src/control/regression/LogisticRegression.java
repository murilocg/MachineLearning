package control.regression;

import org.jblas.DoubleMatrix;

import entity.dto.ConfigGradientDescent;
import entity.model.Regression;
import entity.util.gradient_descent.GradientDescent;
import entity.util.hypoythesis.HypothesisLogisticRegression;

public class LogisticRegression extends Regression {

	public LogisticRegression(double alpha, int iterations) {
		super(alpha, iterations, new HypothesisLogisticRegression());
	}

	@Override
	public void train(DoubleMatrix X, DoubleMatrix Y) {
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		theta = DoubleMatrix.zeros(X.columns, 1);
		ConfigGradientDescent config = new ConfigGradientDescent(X, Y, theta, hypothesis, alpha, iterations);
		theta = GradientDescent.compute(config);
	}
	
	@Override
	public DoubleMatrix predict(DoubleMatrix X) {
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		DoubleMatrix Y = new DoubleMatrix(X.rows, 1);
		for (int i = 0; i < X.rows; i++) {
			Y.put(i, hypothesis.compute(X.getRow(i), theta).get(0));
		}
		return Y;
	}
}
