package entity.util.gradient_descent;

import org.jblas.DoubleMatrix;

import entity.dto.ConfigGradientDescent;

public class GradientDescent {

	public static DoubleMatrix compute(ConfigGradientDescent config) {
		for (int i = 0; i < config.getIterations(); i++) {
			double[] error = calculateThetaErrors(config);
			double[] reg = calculateRegularization(config.getTheta(), config.getLambda());
			DoubleMatrix updatedTheta = updateTheta(config, error, reg);
			config.setTheta(updatedTheta);
		}
		return config.getTheta();
	}

	protected static double[] calculateThetaErrors(ConfigGradientDescent config) {
		int length = config.getTheta().length;
		double[] sums = new double[length];
		for (int j = 0; j < length; j++) {
			sums[j] = computeSum(config, j);
		}
		return sums;
	}

	private static double computeSum(ConfigGradientDescent config, int j) {
		double sum = 0;
		for (int k = 0; k < config.getY().length; k++) {
			DoubleMatrix row = config.getX().getRow(k);
			double h = config.getHypothesis().compute(row, config.getTheta());
			sum += (h - config.getY().get(k)) * row.get(j);
		}
		return sum;
	}

	private static double[] calculateRegularization(DoubleMatrix theta, double lambda) {
		double[] reg = new double[theta.length];
		for (int j = 1; j < theta.length; j++) {
			reg[j] = lambda * theta.get(j);
		}
		return reg;
	}

	private static DoubleMatrix updateTheta(ConfigGradientDescent config, double[] sums, double[] reg) {
		DoubleMatrix theta = config.getTheta();
		for (int j = 1; j < theta.length; j++) {
			double newTheta = computeTheta(theta.get(j), config.getAlpha(), sums[j] + reg[j], config.getY().length);
			theta.put(j, newTheta);
		}
		return theta;
	}

	private static double computeTheta(double theta, double alpha, double sum, int m) {
		return theta - (alpha * sum / m);
	}
}
