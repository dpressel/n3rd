package org.n3rd.util;

import org.n3rd.layers.Layer;
import org.n3rd.layers.TemporalConvolutionalLayer;
import org.n3rd.layers.TemporalConvolutionalLayerBlas;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class TemporalConvolutionalLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {
        Integer nK = (Integer)params.get("nK");
        Integer kL = (Integer)params.get("kL");
        Integer kW = (Integer)params.get("kW");
        Integer embedSz = (Integer)params.get("embedSz");

        if (embedSz == null || embedSz <= 1)
        {
            return new TemporalConvolutionalLayerBlas(nK, kL, kW);
        }
        return new TemporalConvolutionalLayer(nK, kL, kW, embedSz);
    }
}
