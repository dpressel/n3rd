package org.n3rd.util;

import org.n3rd.layers.DropoutLayer;
import org.n3rd.layers.Layer;
import org.sgdtk.CollectionsManip;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class DropoutLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {

        Double p = (Double)params.get("p");
        if (p == null)
        {
            p = 0.5;
        }

        return new DropoutLayer(p);
    }
}
