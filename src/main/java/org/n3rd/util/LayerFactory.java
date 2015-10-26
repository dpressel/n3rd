package org.n3rd.util;

import org.n3rd.layers.Layer;

import java.util.Map;

/**
 * Created by dpressel on 10/24/15.
 */
public interface LayerFactory
{
    Layer newLayer(Map<String, Object> params);
}
