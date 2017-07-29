package entity.util;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;

import org.jblas.DoubleMatrix;

public class LoadData {

	public static DoubleMatrix load(String path, String columnSeparator) {
		DoubleMatrix dataset = null;
		try {
			InputStream is = new FileInputStream(path);
			InputStreamReader isr = new InputStreamReader(is);
			BufferedReader br = new BufferedReader(isr);

			Iterator<String> iterator = br.lines().iterator();
			dataset = getRow(iterator.next(), columnSeparator);
			while (iterator.hasNext()) {
				String line = iterator.next();
				dataset = DoubleMatrix.concatVertically(dataset, getRow(line, columnSeparator));
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dataset;
	}

	private static DoubleMatrix getRow(String line, String columnSeparator) {
		String[] values = line.split(columnSeparator);
		return new DoubleMatrix(1, values.length, parseToDoubleArray(values));
	}

	private static double[] parseToDoubleArray(String[] values) {
		double[] doubleArray = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			double number = 0;
			try {
				number = Double.parseDouble(values[i].trim());
			} catch (NumberFormatException e) {
				e.printStackTrace();
			}
			doubleArray[i] = number;
		}
		return doubleArray;
	}
}
