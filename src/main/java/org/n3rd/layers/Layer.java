package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.VectorN;

/**
 * Contract for a layer
 *
 * Not all layers actually support learning parameters/biases, in which case they simply return null.
 * Forward and backward prop implementations are expected to be implemented and work as intended
 */
public interface Layer
{
    /**
     * Take the input and produce some outputs
     * for a pre-activation layer, this will perform a dot product on each output, and produces a layer that
     * is number of pre-activation units long.  for an activation layer, this will produce exactly the same number of
     * outputs as inputs
     *
     * @param x previous layer inputs or actual inputs
     * @return this layer's outputs
     */

    VectorN forward(VectorN x);

    /**
     * Implement back prop
     *
     * @param chainGrad Deltas from the layer above
     * @param y deltas from this layer
     * @return
     */
    VectorN backward(VectorN chainGrad, double y);
    
    Tensor getParamGrads();
    Tensor getParams();

    double[] getBiasGrads();
    double[] getBiasParams();

}
