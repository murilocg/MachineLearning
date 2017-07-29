package control;

import org.jblas.DoubleMatrix;

import entity.model.LinearRegressionModel;
import entity.util.Hypotesis;

public class LinearRegressionGradientDescent extends LinearRegressionModel {

	private FeatureNormalize normalizer;
	private boolean normalize;
	private double alpha;
	private int iterations;

	public LinearRegressionGradientDescent(double alpha, int iterations, boolean normalize) {
		this.normalize = normalize;
		this.alpha = alpha;
		this.iterations = iterations;
	}
	
	@Override
	public void train(DoubleMatrix X, DoubleMatrix Y) {
		if (normalize) {
			normalizer = new FeatureNormalize(X);
			X = normalizer.normalize();
		}
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		theta = DoubleMatrix.zeros(X.columns, 1);
		theta = GradientDescent.compute(X, Y, theta, alpha, iterations);
	}

	@Override
	public DoubleMatrix predict(DoubleMatrix X) {

		if (normalize)
			X = normalizer.normalizeMatrix(X);

		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);

		DoubleMatrix Y = new DoubleMatrix(X.rows, 1);
		for (int i = 0; i < X.rows; i++) {
			Y.put(i, Hypotesis.compute(X.getRow(i), theta));
		}
		return Y;
	}

}
