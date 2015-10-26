package org.n3rd.util;

import org.n3rd.layers.KMaxPoolingLayer;
import org.n3rd.layers.Layer;
import org.sgdtk.CollectionsManip;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public class KMaxPoolingLayerFactory implements LayerFactory
{
    @Override
    public Layer newLayer(Map<String, Object> params)
    {

        Integer nK = (Integer)params.get("nK");
        Integer embedSz = (Integer)params.get("embedSz");
        Integer k = (Integer)CollectionsManip.getOrDefault(params, "k", new Integer(2));

        return new KMaxPoolingLayer(k, nK, embedSz);
    }
}
