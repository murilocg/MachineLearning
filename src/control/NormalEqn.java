package control;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class NormalEqn {

	public static DoubleMatrix compute(DoubleMatrix X, DoubleMatrix Y) {
		DoubleMatrix XT = X.transpose();
		DoubleMatrix Xmul = XT.mmul(X);
		DoubleMatrix Xinverse = Solve.solve(Xmul, DoubleMatrix.eye(Xmul.rows));
		DoubleMatrix a = Xinverse.mmul(XT);
		return a.mmul(Y);
	}
}
