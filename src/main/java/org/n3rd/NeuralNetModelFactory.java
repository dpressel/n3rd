package org.n3rd;

import org.n3rd.layers.Layer;
import org.n3rd.layers.SigmoidLayer;
import org.sgdtk.Model;
import org.sgdtk.ModelFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Factory for neural net models.
 *
 * Right now this doesnt work as you probably expect, its just fulfilling the required interface since you cannot
 * currently build parallel models with it.  This is easy to fix, but we would want
 * to support then cloning (prototyping) a layer to do so...
 *
 * @author dpressel
 */
public class NeuralNetModelFactory implements ModelFactory
{

    List<Layer> layers;

    public NeuralNetModelFactory()
    {
        layers = new ArrayList<Layer>();
    }
    public NeuralNetModelFactory(Layer[] layers)
    {
        this.layers = new ArrayList<Layer>(Arrays.asList(layers));
    }

    /**
     * Due to inheriting the interface from SGDTk, we have an unused wlength variable, which is safely ignored.
     * @param params Settings, ignored for now but will likely be used heavily in the future
     * @return A NeuralNetModel
     */
    @Override
    public Model newInstance(Object params)
    {
        Layer[] layerArray = layers.toArray(new Layer[layers.size()]);
        boolean scale = false;
        if (layerArray[layerArray.length - 1] instanceof SigmoidLayer)
        {
            scale = true;
        }
        return new NeuralNetModel(layerArray, scale);
    }

    public NeuralNetModelFactory addLayer(Layer layer)
    {
        layers.add(layer);
        return this;
    }
}
