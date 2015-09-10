package org.n3rd.layers;

import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

/**
 * ReLU as its typically described -- max(0, d)
 *
 * @author dpressel
 */
public class ReLULayer extends ActivationLayer
{
    double [] output;

    public ReLULayer()
    {

    }
    //        /
    //       /
    // _____/   x = 0.1, 0.2, 0.3 f(x) = 0.1, 0.2, 0.3 y = 1 * x
    double relu(double d)
    {
        return Math.max(0, d);
    }
    double drelu(double d)
    {
        return d > 0. ? 1.: 0.;
    }
    @Override
    public VectorN forward(VectorN z)
    {

        int sz = z.length();
        output = new double[sz];
        for (int i = 0; i < sz; ++i)
        {
            double zi = z.at(i);
            output[i] = relu(zi);
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
            dh[i] = chainGrad.at(i) * drelu(output[i]);
        }
        return new DenseVectorN(dh);
    }
}
