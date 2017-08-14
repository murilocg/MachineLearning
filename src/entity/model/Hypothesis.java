package entity.model;

import org.jblas.DoubleMatrix;

public interface Hypothesis {
	
	public abstract DoubleMatrix compute(DoubleMatrix X, DoubleMatrix theta);
}
