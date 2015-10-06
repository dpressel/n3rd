package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.CollectionsManip;

/**
 * LogSoftMaxLayer returns outputs in log soft max space
 *
 * @author dpressel
 */
public class LogSoftMaxLayer extends AbstractLayer
{

    @Override
    public Tensor forward(Tensor z)
    {
        int sz = z.size();
        output = new Tensor(sz);
        grads = new Tensor(sz);

        double logsum = CollectionsManip.logSum(z.d);

        for (int i = 0; i < sz; ++i)
        {
            output.d[i] = z.d[i] - logsum;
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        int sz = output.size();
        // Only will be one thing above us, a loss function
        double sum = chainGrad.d[0];

        final int yidx = (int)(y - 1);
        for (int i = 0; i < sz; ++i)
        {
            double indicator = yidx == i ? 1.0: 0.0;
            grads.d[i] = (indicator - Math.exp(output.d[i]))*sum;
        }
        return grads;
    }

}
