package org.n3rd;

import org.n3rd.layers.*;
import org.n3rd.ops.AdagradUpdate;
import org.n3rd.ops.Update;
import org.n3rd.util.*;
import org.sgdtk.Model;
import org.sgdtk.ModelFactory;

import java.util.*;

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

    public static final String TYPE_NAME = "type";
    static Map<String, LayerFactory> layerFactories;

    static {
        layerFactories = new HashMap<String, LayerFactory>();
        layerFactories.put(AverageFoldingLayer.class.getSimpleName(), new AverageFoldingLayerFactory());
        layerFactories.put(KMaxPoolingLayer.class.getSimpleName(), new KMaxPoolingLayerFactory());
        layerFactories.put(TanhLayer.class.getSimpleName(), new TanhLayerFactory());
        layerFactories.put(SigmoidLayer.class.getSimpleName(), new SigmoidLayerFactory());
        layerFactories.put(LogSoftMaxLayer.class.getSimpleName(), new LogSoftMaxLayerFactory());
        layerFactories.put(ReLULayer.class.getSimpleName(), new ReLULayerFactory());
        layerFactories.put(TemporalConvolutionalLayer.class.getSimpleName(), new TemporalConvolutionalLayerFactory());
        layerFactories.put(SpatialConvolutionalLayer.class.getSimpleName(), new SpatialConvolutionalLayerFactory());
        layerFactories.put(MaxPoolingLayer.class.getSimpleName(), new MaxPoolingLayerFactory());
        layerFactories.put(FullyConnectedLayer.class.getSimpleName(), new FullyConnectedLayerFactory());
        layerFactories.put(DropoutLayer.class.getSimpleName(), new DropoutLayerFactory());
        layerFactories.put(MaxOverTimePoolingLayer.class.getSimpleName(), new MaxOverTimePoolingLayerFactory());
    }
    List<Layer> layers;
    private Update update;
/*
    public NeuralNetModelFactory(File file) throws IOException
    {
        NeuralNetModel neuralNetModel = new NeuralNetModel();
        neuralNetModel.load(file);
        layers = new ArrayList(Arrays.asList(neuralNetModel.getLayers()));
    }

    public NeuralNetModelFactory(InputStream inputStream) throws IOException
    {
        NeuralNetModel neuralNetModel = new NeuralNetModel();
        neuralNetModel.load(inputStream);
        layers = new ArrayList(Arrays.asList(neuralNetModel.getLayers()));
    }
*/
    public static void installLayerFactory(String name, LayerFactory layerFactory)
    {
        layerFactories.put(name, layerFactory);
    }
    @Override
    public void configure(Map<String, Object> config) throws Exception
    {

        List<Map<String, Object>> layerConfigs = (List<Map<String, Object>>) config.get("layers");
        layers = new ArrayList<Layer>(layerConfigs.size());
        for (Map<String, Object> layerConfig : layerConfigs)
        {
            LayerFactory layerFactory = layerFactories.get(layerConfig.get(TYPE_NAME));
            layers.add(layerFactory.newLayer(layerConfig));
        }
        Boolean adagradOrNull = (Boolean)config.get("adagrad");
        Double alphaOrNull = (Double)config.get("alpha");

        boolean useAdagrad = adagradOrNull == null ? true: adagradOrNull;

        if (useAdagrad)
        {
            double alpha = alphaOrNull == null ? 1.0 : alphaOrNull;
            setUpdate(new AdagradUpdate(alpha));
        }
        else
        {
            double alpha = alphaOrNull == null ? 0.0 : alphaOrNull;
        }
    }

    public NeuralNetModelFactory()
    {
        this.layers = new ArrayList<Layer>();
    }

    public NeuralNetModelFactory(Layer[] layers)
    {
        this.layers = new ArrayList(Arrays.asList(layers));
    }

    /**
     * Due to inheriting the interface from SGDTk, we have an unused wlength variable, which is safely ignored.
     * @param params Settings, ignored for now but will likely be used heavily in the future
     * @return A NeuralNetModel
     */
    @Override
    public Model newInstance(Object params)
    {

        boolean scale = false;
        if (layers.get(layers.size() - 1) instanceof SigmoidLayer)
        {
            scale = true;
        }

        return new NeuralNetModel(layers.toArray(new Layer[layers.size()]), scale, getUpdate() == null ? new AdagradUpdate(1.0): getUpdate());
    }

    public NeuralNetModelFactory addLayer(Layer layer)
    {
        layers.add(layer);
        return this;
    }

    public Update getUpdate()
    {
        return update;
    }

    public void setUpdate(Update update)
    {
        this.update = update;
    }
}
