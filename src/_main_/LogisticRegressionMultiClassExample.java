package _main_;

import java.text.MessageFormat;

import org.jblas.DoubleMatrix;

import control.FeatureNormalize;
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
		LogisticRegressionMultiClass model = new LogisticRegressionMultiClass(0.001, 2000, classes, 1);
		model.train(X, Y);

		System.out.println("\n\n3º-Predicting...");
		double[] valueToPredict = getValueToPredict();
		DoubleMatrix p = new DoubleMatrix(1, valueToPredict.length, valueToPredict);

		DoubleMatrix prediction = model.predict(p);
		String msg = "In function of inputs we predict a probability of {0} that the number is  {1}";
		msg = MessageFormat.format(msg, prediction.get(1) * 100, prediction.get(0));
		System.out.println(msg);
	}

	private static double[] getValueToPredict() {
		return new double[] { 0, 0, 8, 15, 16, 13, 0, 0, 0, 1, 11, 9, 11, 16, 1, 0, 0, 0, 0, 0, 7, 14, 0, 0, 0, 0, 3, 4,
				14, 12, 2, 0, 0, 1, 16, 16, 16, 16, 10, 0, 0, 2, 12, 16, 10, 0, 0, 0, 0, 0, 2, 16, 4, 0, 0, 0, 0, 0, 9,
				14, 0, 0, 0, 0 };
	}

	private static int[] getArray(int size) {
		int[] arr = new int[size];
		for (int k = 0; k < size; k++) {
			arr[k] = k + 1;
		}
		return arr;
	}
}
