package entity.util;

import org.jblas.DoubleMatrix;

public class BinaryClass {

	public static DoubleMatrix parse(DoubleMatrix y, int klass) {
		DoubleMatrix bY = new DoubleMatrix(y.rows, y.columns);
		for(int i = 0; i < y.length; i++) {
			if(y.get(i) == klass) {
				bY.put(i, 1);
			}else {
				bY.put(i, 0);
			}
		}
		return bY;
	}
}
