package entity.util.cost_function;

import org.jblas.DoubleMatrix;

import entity.model.CostFunction;
import entity.model.Hypothesis;

public class CostFunctionLinearRegression implements CostFunction {

	@Override
	public double cost(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis) {
		DoubleMatrix r1 = hypothesis.compute(x, theta);
		DoubleMatrix negativeY = y.dup().mmuli(-1);
		r1.subi(negativeY);
		r1 = r1.transpose().mmul(r1);
		r1.divi(2 * y.length);
		return r1.get(0);
		// double squaredDifference = computeSquaredDifference(x, y, theta, hypothesis);
		// return squaredDifference / (2 * y.length);
	}

	@Override
	public double costRegularized(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis,
			double lambda) {
		double cost = cost(x, y, theta, hypothesis);
		double reg = reg(theta, lambda, y.length);
		return cost + reg;
	}

	private double reg(DoubleMatrix theta, double lambda, int m) {
		DoubleMatrix newTheta = theta.dup().transpose();
		newTheta.mmul(theta);
		newTheta.mmuli(lambda / (2 * m));
		return newTheta.get(0);
	}
	// double sum = 0;
	// for (int i = 0; i < theta.length; i++) {
	// sum += Math.pow(theta.get(i), 2);
	// }
	// return lambda * sum / m;

	// private double computeSquaredDifference(DoubleMatrix x, DoubleMatrix y,
	// DoubleMatrix theta, Hypothesis hypothesis) {
	// double sum = 0;
	// for (int i = 0; i < y.length; i++) {
	// double h = hypothesis.compute(x.getRow(i), theta);
	// sum += Math.pow(h - y.get(i), 2);
	// }
	// return sum;
	// }
}
