package _main_;

import java.text.MessageFormat;

import org.jblas.DoubleMatrix;

import control.regression.LogisticRegressionMultiClass;
import entity.util.LoadData;

/*
 *  Observation: Logistic Regression for Multivariate class will be 
 *  inefficient with large numbers of features. In the example below 
 *  the number of features is 65. Consequently to train this model 
 *  cost to much time. So for small and medium numbers of features 
 *  this algorithm can be used. But for large numbers of features 
 *  is recommended the use of Neural Networks.
 */
public class LogisticRegressionMultiClassExample {

	public static void main(String[] args) {
		System.out.println("1º-Preprocessing Data...");
		System.out.println("\n	Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/optdigits.tra", ",");
		DoubleMatrix X = dataset.getColumns(getArray(dataset.columns - 1));
		DoubleMatrix Y = dataset.getColumn(dataset.columns - 1);

		System.out.println("\n\n2º-Training Model...");
		System.out.println("\n	Running Logistic Regression Multi Class....");
		int[] classes = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
		LogisticRegressionMultiClass model = new LogisticRegressionMultiClass(0.1, 2000, classes);
		model.train(X, Y);

		System.out.println("\n\n3º-Predicting...");
		double[] valueToPredict = new double[] { 0, 1, 6, 15, 12, 1, 0, 0, 0, 7, 16, 6, 6, 10, 0, 0, 0, 8, 16, 2, 0, 11,
				2, 0, 0, 5, 16, 3, 0, 5, 7, 0, 0, 7, 13, 3, 0, 8, 7, 0, 0, 4, 12, 0, 1, 13, 5, 0, 0, 0, 14, 9, 15, 9, 0,
				0, 0, 0, 6, 14, 7, 1, 0, 0 };
		DoubleMatrix prediction = model.predict(new DoubleMatrix(1, 64, valueToPredict));
		String msg = "In function of inputs we predict a probability of {0} that the number is  {1}";
		msg = MessageFormat.format(msg, prediction.get(1) * 100, prediction.get(0));
		System.out.println(msg);
	}

	private static int[] getArray(int size) {
		int[] arr = new int[size];
		for (int k = 0; k < size; k++) {
			arr[k] = k + 1;
		}
		return arr;
	}
}
