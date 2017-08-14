package entity.util.cost_function;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import entity.model.CostFunction;
import entity.model.Hypothesis;

public class CostFunctionLogisticRegression implements CostFunction {

	@Override
	public double cost(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis) {
		DoubleMatrix sigmoid = hypothesis.compute(x, theta);
		DoubleMatrix sigmoid2 = sigmoid.dup();
		DoubleMatrix negativey = y.dup().muli(-1).transpose();

		// −y^T * log(h)
		MatrixFunctions.logi(sigmoid);
		sigmoid = negativey.mmul(sigmoid);
		sigmoid.divi(Math.log(2));

		// (1 - y^i)
		negativey.addi(1);

		// log(1 − h)
		sigmoid2.muli(-1);
		sigmoid2.addi(1);
		MatrixFunctions.logi(sigmoid2);
		sigmoid.divi(Math.log(2));

		sigmoid2 = negativey.mmul(sigmoid2);

		sigmoid = sigmoid.subi(sigmoid2).divi(y.length);
		return sigmoid.get(0);
	}

	@Override
	public double costRegularized(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis,
			double lambda) {
		double sum = cost(x, y, theta, hypothesis);
		double r = reg(theta, lambda, y.length);
		return (sum + r);
	}

	private static double reg(DoubleMatrix theta, double lambda, int m) {
		DoubleMatrix newTheta = theta.dup().transpose();
		newTheta.put(0, 0);
		newTheta = newTheta.mmul(theta);
		newTheta.mmuli(lambda / (2 * m));
		return newTheta.get(0);
	}
}
