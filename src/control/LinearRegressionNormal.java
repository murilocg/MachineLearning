package control;

import org.jblas.DoubleMatrix;

import entity.model.LinearRegressionModel;

public class LinearRegressionNormal extends LinearRegressionModel {

	@Override
	public void train(DoubleMatrix X, DoubleMatrix Y) {
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		theta = NormalEqn.compute(X, Y);
	}

	@Override
	public DoubleMatrix predict(DoubleMatrix X) {
		X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(X.rows, 1), X);
		return X.mmul(theta);
	}
}
