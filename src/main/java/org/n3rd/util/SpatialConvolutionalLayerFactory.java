package org.n3rd.util;

import org.n3rd.layers.Layer;
import org.n3rd.layers.SpatialConvolutionalLayer;
import org.n3rd.layers.SpatialConvolutionalLayerBlas;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class SpatialConvolutionalLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {

        Integer kH = (Integer)params.get("kH");
        Integer kW = (Integer)params.get("kW");
        Integer kL = (Integer)params.get("kL");
        Integer h = (Integer)params.get("h");
        Integer w = (Integer)params.get("w");
        Integer nK = (Integer)params.get("nK");
        
        return new SpatialConvolutionalLayerBlas(nK, kH, kW, kL, h, w);
    }
}
