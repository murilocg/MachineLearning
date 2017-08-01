package entity.util;

import org.jblas.DoubleMatrix;

import entity.model.Hypothesis;

public class HypothesisLinearRegression implements Hypothesis{

	public double compute(DoubleMatrix x, DoubleMatrix theta) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += (x.get(i) * theta.get(i));
		}
		return sum;
	}
}
