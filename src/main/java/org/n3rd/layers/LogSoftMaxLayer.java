package org.n3rd.layers;

import org.sgdtk.CollectionsManip;
import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

/**
 * LogSoftMaxLayer returns outputs in log soft max space
 *
 * @author dpressel
 */
public class LogSoftMaxLayer extends ActivationLayer
{

    double[] output;
    double[] grad;

    @Override
    public VectorN forward(VectorN z)
    {
        int sz = z.length();
        output = new double[sz];
        grad = new double[sz];

        DenseVectorN denseVectorN = (DenseVectorN)z;
        double[] x = denseVectorN.getX();
        double logsum = CollectionsManip.logSum(x);

        for (int i = 0; i < x.length; ++i)
        {
            output[i] = x[i] - logsum;
        }
        return new DenseVectorN(output);
    }

    @Override
    public VectorN backward(VectorN chainGrad, double y)
    {
        int sz = output.length;
        // Only will be one thing above us, a loss function
        double sum = chainGrad.at(0);

        final int yidx = (int)(y - 1);
        for (int i = 0; i < sz; ++i)
        {
            double indicator = yidx == i ? 1.0: 0.0;
            grad[i] = (indicator - Math.exp(output[i]))*sum;
        }
        return new DenseVectorN(grad);
    }
}
