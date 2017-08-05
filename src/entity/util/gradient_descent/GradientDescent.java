package entity.util.gradient_descent;

import org.jblas.DoubleMatrix;

import entity.dto.ConfigGradientDescent;

public class GradientDescent {

	public static DoubleMatrix compute(ConfigGradientDescent config) {
		for (int i = 0; i < config.getIterations(); i++) {
			double[] sums = calculateThetaErrors(config);
			config.setTheta(updateTheta(config, sums));
		}
		return config.getTheta();
	}

	private static double[] calculateThetaErrors(ConfigGradientDescent config) {
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

	private static DoubleMatrix updateTheta(ConfigGradientDescent config, double[] sums) {
		DoubleMatrix theta = config.getTheta();
		for (int j = 0; j < theta.length; j++) {
			theta.put(j, theta.get(j) - ((config.getAlpha() * sums[j]) / config.getY().length));
		}
		return theta;
	}

}
