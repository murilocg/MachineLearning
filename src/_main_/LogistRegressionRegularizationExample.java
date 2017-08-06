package _main_;

import java.text.MessageFormat;

import org.jblas.DoubleMatrix;

import control.PolynomialFeatureGenerator;
import control.regression.LogisticRegression;
import entity.dto.ConfigGradientDescent;
import entity.model.Hypothesis;
import entity.model.LogisticRegressionModel;
import entity.util.LoadData;
import entity.util.gradient_descent.GradientDescentRegularization;
import entity.util.hypoythesis.HypothesisLogisticRegression;

public class LogistRegressionRegularizationExample {

	public static void main(String[] args) {

		System.out.println("1º-Processing Data");
		System.out.println("\n	Loading Data...");
		DoubleMatrix dataset = LoadData.load("train_data/microchips_quality_assurance.txt", ",");
		DoubleMatrix X = dataset.getColumns(new int[] { 0, 1 });
		DoubleMatrix Y = dataset.getColumn(2);

		System.out.println("\n	Adding polynomial features....");
		X = PolynomialFeatureGenerator.mapFeature(X.getColumn(0), X.getColumn(1), 6);
		DoubleMatrix theta = DoubleMatrix.zeros(X.columns, 1);

		System.out.println("\n\n2º-Training Model....");
		System.out.println("\n	Running Logistic Regression with Regularization to prevent overfittings....");
		Hypothesis hypothesis = new HypothesisLogisticRegression();
		theta = GradientDescentRegularization.compute(new ConfigGradientDescent(X, Y, theta, hypothesis, 0.1, 0.1, 2000));

		System.out.println("\n\n3º-Predicting....");
		DoubleMatrix p = new DoubleMatrix(1, 2, new double[] { 0.051267, 0.69956 });
		p = PolynomialFeatureGenerator.mapFeature(p.getColumn(0), p.getColumn(1), 6);
		String msg = "\n	For a chip with scores {0} and {1}, we predict the chip will be {2} ";
		System.out.println(MessageFormat.format(msg, p.get(1), p.get(2), Math.round(hypothesis.compute(p, theta)) == 1 ? "approved" : "disapproved"));
	}
}