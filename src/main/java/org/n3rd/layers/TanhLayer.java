package org.n3rd.layers;

import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

/**
 * Standard Tanh implementation
 *
 * @author dpressel
 */
public class TanhLayer extends ActivationLayer
{
    double [] output;

    public TanhLayer()
    {

    }

    @Override
    public VectorN forward(VectorN z)
    {

        int sz = z.length();
        output = new double[sz];
        for (int i = 0; i < sz; ++i)
        {
            double zi = z.at(i);
            output[i] = Math.tanh(zi);
        }
        return new DenseVectorN(output);
    }


    @Override
    public VectorN backward(VectorN chainGrad, double y)
    {
        int sz = chainGrad.length();
        double [] dh = new double[sz];
        for (int i = 0; i < sz; ++i)
        {
            dh[i] = chainGrad.at(i) * (1 - output[i]*output[i]);
        }
        return new DenseVectorN(dh);
    }
}
