package entity.util.cost_function;

import org.jblas.DoubleMatrix;

import entity.model.CostFunction;
import entity.model.Hypothesis;

public class CostFunctionLogisticRegression implements CostFunction {

	@Override
	public double compute(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis) {
		double sum = 0;
		for (int i = 0; i < y.rows; i++) {
			double hRow = hypothesis.compute(x.getRow(i), theta);
			double t1 = y.get(i) * Math.log(hRow);
			double t2 = (1 - y.get(i)) * Math.log(1 - hRow);
			sum += (t1 + t2);
		}
		return sum / y.length;
	}

	@Override
	public double compute(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis, double lambda) {
		double sum = 0;
		for (int i = 0; i < y.rows; i++) {
			sum += Math.pow(hypothesis.compute(x.getRow(i), theta) - y.get(i), 2);
		}
		return (sum + regularization(theta, lambda)) / (2 * y.length);
	}

	private double regularization(DoubleMatrix theta, double lambda) {
		double sum = 0;
		for (int i = 1; i < theta.length; i++) {
			sum += Math.pow(theta.get(i), 2);
		}
		return lambda * sum;
	}
}
