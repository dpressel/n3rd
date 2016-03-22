package org.n3rd.layers;

import org.n3rd.Tensor;

import java.util.BitSet;

/**
 * Created by dpressel on 10/26/15.
 */
public class DropoutLayer extends AbstractLayer implements DiffersOnTraining
{

    double probDropTrain;
    double probDrop;

    BitSet bits = new BitSet(1024);

    protected Tensor grads;
    protected Tensor output;

    boolean bernoulli()
    {
        return Math.random() < probDrop;
    }
    public DropoutLayer()
    {
        this(0.0);
    }
    public DropoutLayer(double p)
    {

        this.probDropTrain = probDrop = p;
        output = new Tensor(1);
        grads = new Tensor(1);
    }

    @Override
    public void setIsTraining(boolean training)
    {
        probDrop = training ? probDropTrain: 0;
    }

    @Override
    public Tensor forward(Tensor x)
    {
        try
        {


            if (probDrop > 0.)
            {
                this.output.resize(x.size());
                this.output.reshape(x.dims);
                double scale = 1. / (1. - probDrop);
                int sz = x.size();
                if (bits.size() < x.size())
                {
                    bits = new BitSet(sz);
                }
                else
                {
                    bits.clear();
                }
                for (int i = 0; i < sz; ++i)
                {
                    boolean mask = bernoulli();
                    bits.set(i, mask);
                    output.set(i, mask ? 0. : x.get(i) * scale);

                }

            }
            else
            {
                x.copyTo(output);
            }
            return output;
        }
        catch (Exception ex)
        {
            throw new RuntimeException(ex);
        }

    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        try
        {
            double scale = 1. / (1. - probDrop);
            int sz = chainGrad.size();
            grads.resize(sz);
            grads.reshape(chainGrad.dims);
            grads.constant(0.);

            for (int i = 0; i < sz; ++i)
            {
                grads.set(i, bits.get(i) ? 0. : chainGrad.get(i) * scale);
            }
            return grads;
        }
        catch (Exception ex)
        {
            throw new RuntimeException(ex);
        }

    }

}
