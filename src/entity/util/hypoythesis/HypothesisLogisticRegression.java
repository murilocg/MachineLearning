package entity.util.hypoythesis;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import entity.model.Hypothesis;


public class HypothesisLogisticRegression implements Hypothesis {
	
	
	public DoubleMatrix compute(DoubleMatrix x, DoubleMatrix theta) {
		DoubleMatrix matrix_z = x.mmul(theta);
		matrix_z = matrix_z.mmuli(-1);
		DoubleMatrix ones = DoubleMatrix.ones(matrix_z.rows, matrix_z.columns);
		ones.mmuli(Math.E);
		ones = MatrixFunctions.powi(ones, matrix_z);
		ones.addi(1);
		return  MatrixFunctions.powi(ones, -1);
	}
}
