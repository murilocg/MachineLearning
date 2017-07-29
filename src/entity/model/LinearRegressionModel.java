package entity.model;

import org.jblas.DoubleMatrix;

public abstract class LinearRegressionModel {

	protected DoubleMatrix theta;

	public abstract void train(DoubleMatrix X, DoubleMatrix Y);

	public abstract DoubleMatrix predict(DoubleMatrix X);
}
