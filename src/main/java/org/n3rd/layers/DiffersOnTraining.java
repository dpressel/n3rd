package org.n3rd.layers;

import org.n3rd.Tensor;

/**
 * Contract for a layer
 *
 * Not all layers actually support learning parameters/biases, in which case they simply return null.
 * Forward and backward prop implementations are expected to be implemented and work as intended
 */
public interface DiffersOnTraining
{

    void setIsTraining(boolean isTraining);
}
