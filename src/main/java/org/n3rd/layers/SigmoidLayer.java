package org.n3rd.layers;

import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

/**
 *  Standard sigmoid layer
 *
 *  @author dpressel
 */
public class SigmoidLayer extends ActivationLayer
{
    public SigmoidLayer()
    {

    }
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    double[] output;
    double[] grad;


    // We got pre-activations out of the last layer.  All we are doing here is squashing them, forming
    // activations.  That means that we have the same number of outputs as we have inputs!
    @Override
    public VectorN forward(VectorN z)
    {
        int sz = z.length();
        output = new double[sz];
        grad = new double[sz];
        for (int i = 0; i < sz; ++i)
        {
            output[i] = sigmoid(z.at(i));
        }
        return new DenseVectorN(output);
    }

    @Override
    public VectorN backward(VectorN chainGrad, double y)
    {
        for (int i = 0, sz = chainGrad.length(); i < sz; ++i)
        {
            grad[i] = chainGrad.at(i) * (1 - output[i]) * output[i];
        }
        return new DenseVectorN(grad);
    }
}
