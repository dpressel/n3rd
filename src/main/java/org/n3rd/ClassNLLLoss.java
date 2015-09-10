package org.n3rd;

import org.sgdtk.Loss;

/**
 * Similar in concept to torch, Im assuming since both are probably drawn from similar origins
 * We are going to pass this through backprop to a softmax layer is the assumption.
 *
 * @author dpressel
 */
public class ClassNLLLoss implements Loss
{
    /**
     * We know that this is a multiclass layer, so p should be the max probability, allowing us to return the NLL
     * @param p
     * @param y
     * @return
     */
    @Override
    public double loss(double p, double y)
    {
        return -p;
    }

    @Override
    public double dLoss(double p, double y)
    {
        return -1;
    }
}
