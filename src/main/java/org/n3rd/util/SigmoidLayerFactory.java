package org.n3rd.util;

import org.n3rd.layers.Layer;
import org.n3rd.layers.SigmoidLayer;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class SigmoidLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {
        return new SigmoidLayer();
    }
}
