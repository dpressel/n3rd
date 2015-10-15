package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

/**
 * ReLU as its typically described -- max(0, d)
 *
 * @author dpressel
 */
public class ReLULayer extends AbstractLayer
{

    public ReLULayer()
    {
        output = new Tensor(1);
        grads = new Tensor(1);
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
    public Tensor forward(Tensor z)
    {

        final int sz = z.size();
        output.resize(sz);
        grads.resize(sz);

        ArrayDouble oA = output.getArray();
        for (int i = 0; i < sz; ++i)
        {
            double zi = z.at(i);
            oA.set(i, relu(zi));
        }
        return output;
    }


    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        final int sz = chainGrad.size();

        ArrayDouble gA = grads.getArray();
        for (int i = 0; i < sz; ++i)
        {
            gA.set(i, chainGrad.at(i) * drelu(output.at(i)));
        }
        return grads;
    }
}
