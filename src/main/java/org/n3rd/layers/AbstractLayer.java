package org.n3rd.layers;

import org.n3rd.Tensor;

/**
 * Provide a base layer general enough for most implementations
 * @author dpressel
 */
public abstract class AbstractLayer implements Layer
{

    protected double[] biasGrads;
    protected double[] biases;
    protected Tensor weights;
    protected Tensor grads;
    protected Tensor gradsW;
    protected Tensor output;

    @Override
    public Tensor getParamGrads()
    {
        return gradsW;
    }

    @Override
    public Tensor getParams()
    {
        return weights;
    }

    @Override
    public double[] getBiasGrads()
    {
        return biasGrads;
    }

    @Override
    public double[] getBiasParams()
    {
        return biases;
    }

    @Override
    public Tensor getOutput()
    {
        return grads;
    }

}
