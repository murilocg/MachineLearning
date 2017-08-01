package control.regression;

import org.jblas.DoubleMatrix;

import control.FeatureNormalize;
import control.GradientDescent;
import entity.model.Hypothesis;
import entity.model.LinearRegressionModel;
import entity.util.HypothesisLinearRegression;

public class LinearRegressionGradientDescent extends LinearRegressionModel {

	private FeatureNormalize normalizer;
	private boolean normalize;
	private double alpha;
	private int iterations;
	private Hypothesis hypothesis;

	public LinearRegressionGradientDescent(double alpha, int iterations, boolean normalize) {
		this.normalize = normalize;
		this.alpha = alpha;
		this.iterations = iterations;
		this.hypothesis = new HypothesisLinearRegression();
	}
	
	@Override
	public void train(DoubleMatrix X, DoubleMatrix Y) {
		if (normalize) {
			normalizer = new FeatureNormalize(X);
			X = normalizer.normalize();
		}
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		theta = DoubleMatrix.zeros(X.columns, 1);
		theta = GradientDescent.compute(X, Y, theta, alpha, iterations, hypothesis);
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
