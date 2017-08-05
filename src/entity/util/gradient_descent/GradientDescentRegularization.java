package entity.util.gradient_descent;

import org.jblas.DoubleMatrix;

import entity.dto.ConfigGradientDescent;

public class GradientDescentRegularization {

	public static DoubleMatrix compute(ConfigGradientDescent config) {
		for (int i = 0; i < config.getIterations(); i++) {
			double[] error = calculateThetaErrors(config);
			config.setTheta(updateTheta(config, error));
		}
		return config.getTheta();
	}

	private static double[] calculateThetaErrors(ConfigGradientDescent config) {
		int length = config.getTheta().length;
		double[] error = new double[length];
		error[0] = calculateError(config, 0);
		for (int j = 1; j < length; j++) {
			error[j] = calculateError(config, j);
		}
		return error;
	}

	private static double calculateError(ConfigGradientDescent config, int j) {
		double sum = 0;
		for (int k = 0; k < config.getY().length; k++) {
			DoubleMatrix row = config.getX().getRow(k);
			double h = config.getHypothesis().compute(row, config.getTheta());
			sum += (h - config.getY().get(k)) * row.get(j);
		}
		return sum;
	}

	private static DoubleMatrix updateTheta(ConfigGradientDescent config, double[] error) {
		DoubleMatrix theta = config.getTheta();
		for (int j = 0; j < theta.length; j++) {
			double r = config.getLambda() * theta.get(j);
			theta.put(j, theta.get(j) - config.getAlpha() * (error[j] + r) / config.getY().length);
		}
		return theta;
	}
}
