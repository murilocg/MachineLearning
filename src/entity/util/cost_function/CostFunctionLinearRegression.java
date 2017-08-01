package entity.util.cost_function;

import org.jblas.DoubleMatrix;

import entity.model.CostFunction;
import entity.model.Hypothesis;

public class CostFunctionLinearRegression implements CostFunction {

	@Override
	public double compute(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis) {
		double squaredDifference = computeSquaredDifference(x, y, theta, hypothesis);
		return squaredDifference / (2 * y.length);
	}

	private double computeSquaredDifference(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis) {
		double sum = 0;
		for (int i = 0; i < y.length; i++) {
			double h = hypothesis.compute(x.getRow(i), theta);
			sum += Math.pow(h - y.get(i), 2);
		}
		return sum;
	}
}
