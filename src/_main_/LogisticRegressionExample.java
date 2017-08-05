package _main_;

import org.jblas.DoubleMatrix;

import control.regression.LogisticRegression;
import entity.util.LoadData;

public class LogisticRegressionExample {

	public static void main(String[] args) {

		System.out.println("Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/students_admissions.txt", ",");
		DoubleMatrix X = dataset.getColumns(new int[] { 0, 1 });
		DoubleMatrix Y = dataset.getColumn(dataset.columns - 1);

		System.out.println("\n\nRunning Logistic Regression with Gradient Descent....");
		LogisticRegression model = new LogisticRegression(0.001, 300000);
		model.train(X, Y);

		System.out.println("\n\nPredicting Prices....");
		DoubleMatrix prediction = model.predict(new DoubleMatrix(1, 2, new double[] { 45, 85 }));
		System.out.println("For a student with scores 45 and 85, we predict an admission probability of: "
				+ (prediction.get(0) * 100));
	}
}
