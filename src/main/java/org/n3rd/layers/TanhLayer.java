package org.n3rd.layers;

import org.n3rd.Tensor;

/**
 * Standard Tanh implementation
 *
 * @author dpressel
 */
public class TanhLayer extends AbstractLayer
{

    public TanhLayer()
    {

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
            output.d[i] = Math.tanh(zi);
        }
        return output;
    }


    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        int sz = chainGrad.size();

        for (int i = 0; i < sz; ++i)
        {
            grads.d[i] = chainGrad.d[i] * (1 - output.d[i]*output.d[i]);
        }
        return grads;
    }

}
