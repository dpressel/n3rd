package org.n3rd.layers;

import org.n3rd.Tensor;

/**
 *  Standard sigmoid layer
 *
 *  @author dpressel
 */
public class SigmoidLayer extends AbstractLayer
{
    public SigmoidLayer()
    {

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
        int sz = z.size();

        output = new Tensor(sz);
        grads = new Tensor(sz);
        for (int i = 0; i < sz; ++i)
        {
            output.d[i] = sigmoid(z.d[i]);
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        for (int i = 0, sz = chainGrad.size(); i < sz; ++i)
        {
            grads.d[i] = chainGrad.d[i] * (1 - output.d[i]) * output.d[i];
        }
        return grads;
    }

}
