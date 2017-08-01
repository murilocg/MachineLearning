package entity.model;

import org.jblas.DoubleMatrix;

public interface CostFunction {

	public abstract double compute(DoubleMatrix x, DoubleMatrix y, DoubleMatrix theta, Hypothesis hypothesis);

}
