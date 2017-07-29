package control;

import org.jblas.DoubleMatrix;

import entity.util.Hypotesis;

public class CostFunction {

	public static double compute(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta) {
		double squaredDifference = computeSquaredDifference(x, y, theta);
		return squaredDifference / (2 * y.length);
	}

	private static double computeSquaredDifference(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta) {
		double sum = 0;
		for (int i = 0; i < y.length; i++) {
			double hypotesis = Hypotesis.compute(x.getRow(i), theta);
			sum += Math.pow(hypotesis - y.get(i), 2);
		}
		return sum;
	}
}
