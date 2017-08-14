package entity.util.gradient_descent;

import org.jblas.DoubleMatrix;

import entity.dto.ConfigGradientDescent;
import entity.model.Hypothesis;

public class GradientDescent {

	public static DoubleMatrix compute(ConfigGradientDescent config) {
		DoubleMatrix X = config.getX();
		DoubleMatrix Y = config.getY();
		DoubleMatrix theta = config.getTheta();
		Hypothesis hypothesis = config.getHypothesis();
		double lambda = config.getLambda();
		double alpha = config.getAlpha();
		for (int i = 0; i < config.getIterations(); i++) {
			DoubleMatrix derivateJ = partialDerivative(X, theta, Y, hypothesis);
			DoubleMatrix reg = reg(theta, lambda, Y.length);
			derivateJ.addi(reg);
			derivateJ.mmuli(alpha);
			theta.subi(derivateJ);
		}
		return theta;
	}

	private static DoubleMatrix reg(DoubleMatrix theta, double lambda, int m) {
		DoubleMatrix r = theta.dup();
		r.put(0, 0);
		r.mmuli(lambda / m);
		return r;
	}

	private static DoubleMatrix partialDerivative(DoubleMatrix X, DoubleMatrix theta, DoubleMatrix Y,
			Hypothesis hypothesis) {
		DoubleMatrix vetorizeComputation = hypothesis.compute(X, theta);
		vetorizeComputation.subi(Y);
		DoubleMatrix XT = X.transpose();
		DoubleMatrix r = XT.mmul(vetorizeComputation);
		r.divi(Y.length);
		return r;
	}
}
