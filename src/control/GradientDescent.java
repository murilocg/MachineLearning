package control;

import org.jblas.DoubleMatrix;

import entity.model.Hypothesis;

public class GradientDescent {

	public static DoubleMatrix compute(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, double alpha, int iterations,
			Hypothesis hypothesis) {
		for (int i = 0; i < iterations; i++) {
			double[] sums = new double[theta.length];
			for (int j = 0; j < theta.length; j++) {
				sums[j] = computeSum(x, y, theta, j, hypothesis);
			}
			theta = updateTheta(y, theta, alpha, sums);
		}
		return theta;
	}

	private static DoubleMatrix updateTheta(DoubleMatrix y, DoubleMatrix theta, double alpha, double[] sums) {
		for (int j = 0; j < theta.length; j++) {
			theta.put(j, theta.get(j) - (alpha / y.length * sums[j]));
		}
		return theta;
	}

	private static double computeSum(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, int j, Hypothesis hypothesis) {
		double sum = 0;
		for (int k = 0; k < y.length; k++) {
			DoubleMatrix row = x.getRow(k);
			double h = hypothesis.compute(row, theta);
			sum += (h - y.get(k)) * row.get(j);
		}
		return sum;
	}

}
