package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

/**
 *  Standard sigmoid layer
 *
 *  @author dpressel
 */
public class SigmoidLayer extends AbstractLayer
{
    public SigmoidLayer()
    {
        output = new Tensor(1);
        grads = new Tensor(1);
    }
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // We got pre-activations out of the last layer.  All we are doing here is squashing them, forming
    // activations.  That means that we have the same number of outputs as we have inputs!
    @Override
    public Tensor forward(Tensor z)
    {
        final int sz = z.size();
        output.resize(sz);
        grads.resize(sz);
        ArrayDouble oA = output.getArray();

        for (int i = 0; i < sz; ++i)
        {
            oA.set(i, sigmoid(z.get(i)));
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {


        ArrayDouble gA = grads.getArray();

        for (int i = 0, sz = chainGrad.size(); i < sz; ++i)
        {
            double oi = output.at(i);
            gA.set(i, chainGrad.at(i) * (1 - oi) * oi);
        }
        return grads;
    }

}
