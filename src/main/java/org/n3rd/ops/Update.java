package org.n3rd.ops;

import org.n3rd.layers.Layer;

/**
 * Created by dpressel on 4/7/16.
 */
public interface Update
{
    void run(Layer layer, double eta, double lambda);

}
