package entity.util.hypoythesis;

import org.jblas.DoubleMatrix;

import entity.model.Hypothesis;

public class HypothesisLinearRegression implements Hypothesis {

	@Override
	public DoubleMatrix compute(DoubleMatrix x, DoubleMatrix theta) {
		return x.mmul(theta);
	}
}
