package org.n3rd.util;

import org.n3rd.layers.Layer;
import org.n3rd.layers.MaxOverTimePoolingLayer;


import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class MaxOverTimePoolingLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {


        Integer kL = (Integer)params.get("kL");
        return new MaxOverTimePoolingLayer(kL);
    }
}
