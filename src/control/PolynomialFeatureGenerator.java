package control;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class PolynomialFeatureGenerator {
	
	public static DoubleMatrix mapFeature(DoubleMatrix X1, DoubleMatrix X2, double degree) {
		DoubleMatrix X = DoubleMatrix.ones(X1.rows, 1);
		for (int i = 1; i <= degree; i++) {
			for (int j = 0; j <= i; j++) {
				DoubleMatrix NX = MatrixFunctions.pow(X1, i - j).mul(MatrixFunctions.pow(X2, j));
				X = DoubleMatrix.concatHorizontally(X, NX);
			}
		}
		return X;
	}
}
