package entity.model;

import org.jblas.DoubleMatrix;

public interface Hypothesis {
	
	public abstract double compute(DoubleMatrix X, DoubleMatrix theta);

}
