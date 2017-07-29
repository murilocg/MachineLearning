package entity.util;

import org.jblas.DoubleMatrix;

public class Statistic {

	public static double std(DoubleMatrix column, double mean) {
		double sumSquaredDifference = 0;
		for (int i = 0; i < column.length; i++) {
			sumSquaredDifference += Math.pow(column.get(i) - mean, 2);
		}
		return Math.sqrt(sumSquaredDifference / column.length);
	}

	public static double std(DoubleMatrix column) {
		double sumSquaredDifference = 0;
		double mean = column.mean();
		for (int i = 0; i < column.length; i++) {
			sumSquaredDifference += Math.pow(column.get(i) - mean, 2);
		}
		return Math.sqrt(sumSquaredDifference / column.length);
	}
}
