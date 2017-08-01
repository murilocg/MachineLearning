package entity.util;

import org.jblas.DoubleMatrix;

import entity.model.Hypothesis;

public class HypothesisLogisticRegression implements Hypothesis {

	public double compute(DoubleMatrix x, DoubleMatrix theta) {
		DoubleMatrix matrix_z = x.mmul(theta);
		double z = matrix_z.get(0);
		return 1 / (1 + Math.pow(Math.E, -z));
	}
}
