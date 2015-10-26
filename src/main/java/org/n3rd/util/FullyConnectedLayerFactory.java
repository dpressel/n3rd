package org.n3rd.util;

import org.n3rd.layers.FullyConnectedLayer;
import org.n3rd.layers.Layer;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class FullyConnectedLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {

        Integer outputLength = (Integer)params.get("outputLength");
        Integer inputLength = (Integer)params.get("inputLength");

        return new FullyConnectedLayer(outputLength, inputLength);
    }
}
