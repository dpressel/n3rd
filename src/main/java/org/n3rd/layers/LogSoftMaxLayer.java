package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;
import org.sgdtk.CollectionsManip;

/**
 * LogSoftMaxLayerFactory returns outputs in log soft max space
 *
 * @author dpressel
 */
public class LogSoftMaxLayer extends AbstractLayer
{

    public LogSoftMaxLayer()
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
        double logsum = CollectionsManip.logSum(z.getArray().v);

        for (int i = 0; i < sz; ++i)
        {
            oA.set(i, z.at(i) - logsum);
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        final int sz = output.size();

        ArrayDouble gA = grads.getArray();

        // Only will be one thing above us, a loss function
        double sum = chainGrad.getArray().at(0);

        final int yidx = (int)(y - 1);
        for (int i = 0; i < sz; ++i)
        {
            double indicator = yidx == i ? 1.0: 0.0;
            gA.set(i, (indicator - Math.exp(output.at(i)))*sum);
        }
        return grads;
    }

}
