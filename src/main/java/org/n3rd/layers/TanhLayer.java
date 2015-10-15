package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

/**
 * Standard Tanh implementation
 *
 * @author dpressel
 */
public class TanhLayer extends AbstractLayer
{

    public TanhLayer()
    {
        output = new Tensor(1);
        grads = new Tensor(1);
    }

    @Override
    public Tensor forward(Tensor z)
    {

        int sz = z.size();
        output.resize(sz);
        grads.resize(sz);

        ArrayDouble oA = output.getArray();
        for (int i = 0; i < sz; ++i)
        {
            double zi = z.at(i);
            oA.set(i, Math.tanh(zi));
        }
        return output;
    }


    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        ArrayDouble gA = grads.getArray();
        int sz = chainGrad.size();

        for (int i = 0; i < sz; ++i)
        {
            double oi = output.at(i);
            gA.set(i, chainGrad.at(i) * (1. - oi*oi));
        }
        return grads;
    }

}
