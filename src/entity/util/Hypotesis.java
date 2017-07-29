package entity.util;

import org.jblas.DoubleMatrix;

public class Hypotesis {

	public static double compute(DoubleMatrix x, DoubleMatrix theta) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += (x.get(i) * theta.get(i));
		}
		return sum;
	}
}
