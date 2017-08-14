package entity.util;

import org.jblas.DoubleMatrix;

public class BinaryClass {

	public static DoubleMatrix parse(DoubleMatrix y, int klass) {
		DoubleMatrix bY = y.dup();
		for (int i = 0; i < bY.length; i++) {
			double p = 0;
			if (bY.get(i) != klass) {
				p = 1;
			}
			bY.put(i, p);
		}
		return bY;
	}
}
