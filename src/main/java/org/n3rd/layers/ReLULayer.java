package org.n3rd.layers;

import org.n3rd.Tensor;

/**
 * ReLU as its typically described -- max(0, d)
 *
 * @author dpressel
 */
public class ReLULayer extends AbstractLayer
{

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
    public Tensor forward(Tensor z)
    {

        int sz = z.size();
        output = new Tensor(sz);
        grads = new Tensor(sz);
        for (int i = 0; i < sz; ++i)
        {
            double zi = z.d[i];
            output.d[i] = relu(zi);
        }
        return output;
    }


    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        int sz = chainGrad.size();

        for (int i = 0; i < sz; ++i)
        {
            grads.d[i] = chainGrad.d[i] * drelu(output.d[i]);
        }
        return grads;
    }
}
