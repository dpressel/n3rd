package org.n3rd.layers;

import org.n3rd.Tensor;

/**
 * Base implementation for an activation layer, just says what we *arent* doing
 *
 * An activation layer, non-linearity, squashing function, etc.  This class is not going to support parameters or bias!
 *
 * @author dpressel
 */
public abstract class ActivationLayer implements Layer
{

    @Override
    public Tensor getParamGrads()
    {
        return null;
    }

    @Override
    public Tensor getParams()
    {
        return null;
    }

    @Override
    public double[] getBiasGrads()
    {
        return null;
    }

    @Override
    public double[] getBiasParams()
    {
        return null;
    }

}
