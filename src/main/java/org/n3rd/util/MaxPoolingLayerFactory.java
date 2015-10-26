package org.n3rd.util;

import org.n3rd.layers.Layer;
import org.n3rd.layers.MaxPoolingLayer;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class MaxPoolingLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {

        Integer dh = (Integer)params.get("dh");
        Integer dw = (Integer)params.get("dw");
        Integer h = (Integer)params.get("h");
        Integer w = (Integer)params.get("w");
        Integer kL = (Integer)params.get("kL");

        return new MaxPoolingLayer(dh, dw, kL, h, w);
    }
}
