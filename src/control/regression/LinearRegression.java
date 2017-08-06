package control.regression;

import org.jblas.DoubleMatrix;

import control.FeatureNormalize;
import entity.dto.ConfigGradientDescent;
import entity.model.Regression;
import entity.util.gradient_descent.GradientDescent;
import entity.util.hypoythesis.HypothesisLinearRegression;

public class LinearRegression extends Regression {

	private FeatureNormalize normalizer;
	private boolean normalize;

	public LinearRegression(double alpha, int iterations, boolean normalize) {
		super(alpha, iterations, new HypothesisLinearRegression());
		this.normalize = normalize;
	}

	@Override
	public void train(DoubleMatrix X, DoubleMatrix Y) {
		if (normalize) {
			normalizer = new FeatureNormalize(X);
			X = normalizer.normalize();
		}
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		theta = DoubleMatrix.zeros(X.columns, 1);
		ConfigGradientDescent config = new ConfigGradientDescent(X, Y, theta, hypothesis, alpha, iterations);
		theta = GradientDescent.compute(config);
	}

	@Override
	public DoubleMatrix predict(DoubleMatrix X) {
		if (normalize)
			X = normalizer.normalizeMatrix(X);
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		DoubleMatrix Y = new DoubleMatrix(X.rows, 1);
		for (int i = 0; i < X.rows; i++) {
			Y.put(i, hypothesis.compute(X.getRow(i), theta));
		}
		return Y;
	}

}
