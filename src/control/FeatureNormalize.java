package control;

import org.jblas.DoubleMatrix;

import entity.util.Statistic;

public class FeatureNormalize {

	private DoubleMatrix mu;
	private DoubleMatrix sigma;
	private DoubleMatrix X;

	public FeatureNormalize(DoubleMatrix X) {
		this.X = X;
		this.mu = mean(X);
		this.sigma = std(X);
	}

	public DoubleMatrix normalize() {
		for (int i = 0; i < X.columns; i++) {
			X.putColumn(i, normalizeColumn(X.getColumn(i), i));
		}
		return X;
	}

	public DoubleMatrix normalizeMatrix(DoubleMatrix X) {
		for (int i = 0; i < X.rows; i++) {
			X.putRow(i, normalizeRow(X.getRow(i)));
		}
		return X;
	}

	public DoubleMatrix normalizeRow(DoubleMatrix row) {
		for (int i = 0; i < row.length; i++) {
			row.put(i, normalizeOneValue(row.get(i), i));
		}
		return row;
	}

	private DoubleMatrix normalizeColumn(DoubleMatrix column, int i) {
		for (int j = 0; j < column.length; j++) {
			column.put(j, normalizeOneValue(column.get(j), i));
		}
		return column;
	}

	public double normalizeOneValue(double value, int column) {
		return (value - mu.get(column)) / sigma.get(column);
	}

	private DoubleMatrix std(DoubleMatrix X) {
		DoubleMatrix sigma = new DoubleMatrix(1, X.columns);
		for (int i = 0; i < X.columns; i++) {
			DoubleMatrix column = X.getColumn(i);
			sigma.put(i, Statistic.std(column));
		}
		return sigma;
	}

	private DoubleMatrix mean(DoubleMatrix X) {
		DoubleMatrix mu = new DoubleMatrix(1, X.columns);
		for (int i = 0; i < X.columns; i++) {
			DoubleMatrix column = X.getColumn(i);
			mu.put(i, column.mean());
		}
		return mu;
	}
}
