package entity.dto;

import org.jblas.DoubleMatrix;

import entity.model.Hypothesis;

public class ConfigGradientDescent {

	private DoubleMatrix X;
	private DoubleMatrix Y;
	private DoubleMatrix theta;
	private Hypothesis hypothesis;
	private double alpha;
	private double lambda;
	private int iterations;

	public ConfigGradientDescent(DoubleMatrix X, DoubleMatrix Y, DoubleMatrix theta, Hypothesis hypothesis,
			double alpha, int iterations) {
		this.X = X;
		this.Y = Y;
		this.theta = theta;
		this.hypothesis = hypothesis;
		this.alpha = alpha;
		this.iterations = iterations;
	}

	public ConfigGradientDescent(DoubleMatrix X, DoubleMatrix Y, DoubleMatrix theta, Hypothesis hypothesis,
			double alpha, double lambda, int iterations) {
		this(X, Y, theta, hypothesis, alpha, iterations);
		this.lambda = lambda;
	}

	public DoubleMatrix getX() {
		return X;
	}

	public void setX(DoubleMatrix x) {
		X = x;
	}

	public DoubleMatrix getY() {
		return Y;
	}

	public void setY(DoubleMatrix y) {
		Y = y;
	}

	public DoubleMatrix getTheta() {
		return theta;
	}

	public void setTheta(DoubleMatrix theta) {
		this.theta = theta;
	}

	public Hypothesis getHypothesis() {
		return hypothesis;
	}

	public void setHypothesis(Hypothesis hypothesis) {
		this.hypothesis = hypothesis;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public double getLambda() {
		return lambda;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public int getIterations() {
		return iterations;
	}

	public void setIterations(int iterations) {
		this.iterations = iterations;
	}
}
