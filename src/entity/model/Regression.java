package entity.model;

import org.jblas.DoubleMatrix;

public abstract class Regression {
	
	protected DoubleMatrix theta;
	protected double alpha;
	protected int iterations;
	protected Hypothesis hypothesis;
	
	public Regression(double alpha, int iterations, Hypothesis hypothesis) {
		this.alpha = alpha;
		this.iterations = iterations;
		this.hypothesis = hypothesis;
	}
	
	public abstract void train(DoubleMatrix X, DoubleMatrix Y);

	public abstract DoubleMatrix predict(DoubleMatrix X);

}
