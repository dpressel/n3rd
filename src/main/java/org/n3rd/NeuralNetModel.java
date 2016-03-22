package org.n3rd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.n3rd.layers.DiffersOnTraining;
import org.n3rd.layers.Layer;
import org.n3rd.util.Layers;
import org.sgdtk.*;

import java.io.*;
import java.util.*;

/**
 * Flexible plug-n-play Layered Neural network model trained with Adagrad
 *
 * This model is basically a wrapper around a set of Layers as building blocks, similar to Torch, and trained
 * using Adagrad (for now anyway)
 *
 * For simplicity, serialization is done using Jackson to JSON, though we could use the fast serialization provided
 * in SGDTk in the future if its helpful.
 *
 * @author dpressel
 */
public class NeuralNetModel implements WeightModel
{
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    protected Layer[] layers;
    protected boolean scaleOutput = true;

    private static final double EPS = 1e-8;

    /**
     * Default constructor, needed to reincarnate models
     */
    public NeuralNetModel()
    {
    }

    /**
     * Constructor supporting a stack of layers, and an argument of whether to center the output at zero and scale
     * it between -1 and 1.  This detail is encapsulated with the NeuralNetModelFactory is the preferred way to create a
     * NeuralNetModel
     *
     * @param layers A stack of network layers
     * @param scaleOutput Scale and center data?
     */
    public NeuralNetModel(Layer[] layers, boolean scaleOutput)
    {
        this.layers = new Layer[layers.length];
        for (int i = 0; i < layers.length; ++i)
        {
            this.layers[i] = layers[i];
        }
        this.scaleOutput = scaleOutput;

    }

    public void setIsTraining(boolean isTraining)
    {
        for (Layer layer : layers)
        {
            // e.g. Dropout
            if (layer instanceof DiffersOnTraining)
            {
                DiffersOnTraining dot = (DiffersOnTraining)layer;
                dot.setIsTraining(isTraining);
            }
        }
    }
    // Here ya go!
    public Layer[] getLayers()
    {
        return layers;
    }

    // Leftover mismatch from base model in SGDTk
    @Override
    public double mag()
    {
        return 0;
    }

    void adagradUpdate(Layer layer, double eta, double lambda)
    {
        Tensor gg = layer.getWeightAccum();
        Tensor weights = layer.getParams();
        int wSz = weights.size();

        Tensor weightGrads = layer.getParamGrads();

        for (int i = 0; i < wSz; ++i)
        {
            if (weightGrads.get(i) == 0.0)
            {
                continue;
            }


            double gwi = weightGrads.get(i);
            gg.addi(i, gwi * gwi);
            double ggi = gg.get(i);

            double etaThis = eta / Math.sqrt(ggi + EPS);
            double delta = -etaThis * gwi;

            //double delta = -eta * gwi;
            double wi = weights.get(i) * (1 - eta * lambda);
            wi += delta;
            weights.set(i, wi);
            weightGrads.set(i, 0);

        }
    }

    void updateBiasWeights(Layer layer, double eta)
    {
        double[] biasGrads = layer.getBiasGrads();
        double[] biasParams = layer.getBiasParams();
        for (int i = 0; i < biasParams.length; ++i)
        {
            // Dont bother to regularize
            double delta = -(biasGrads[i] * eta);// * 0.01; // last number is total fudge
            biasParams[i] += delta;
            biasGrads[i] = 0;
        }
    }
    void updateLayerWeights(Layer layer, double eta, double lambda)
    {
        // Now we need to update each layer's weights
        Tensor weights = layer.getParams();

        // Sometimes weights can be NULL in layers without parameters, dont touch them!
        if (weights != null)
        {
            adagradUpdate(layer, eta, lambda);
        }

        double[] biasParams = layer.getBiasParams();

        // Same story for biasParams, can be NULL
        if (biasParams != null && biasParams.length > 0)
        {

            updateBiasWeights(layer, eta);
        }
    }

    /**
     * Function to update the weights for a model.  I realize in retrospect that this may have been better encapsulated
     * in the actual Trainer, but this wouldve caused a coupling between the Trainer and the Model which we have
     * managed to avoid, which is sort of convenient for SGDTk, since, in the case of flatter models,
     * this manifests in a fairly straightforward implementation.  However, for a Neural Net, the result isnt
     * that desirable -- we end up jamming the backprop guts in here, and the type of update itself as well,
     * which means we are doing Adagrad details within here for now.
     *
     * @param vector
     * @param eta
     * @param lambda
     * @param dLoss
     * @param y
     */
    @Override
    public void updateWeights(VectorN vector, double eta, double lambda, double dLoss, double y)
    {

        Tensor chainGrad = new Tensor(new ArrayDouble(1, dLoss), 1);


        for (int k = layers.length - 1; k >= 0; --k)
        {
            Layer layer = layers[k];

            // This updates the entire chain back, which handles our deltas, so now we have the backward delta
            // during this step, the weight params, if they exist should have also been computed
            chainGrad = layer.backward(chainGrad, y);
            updateLayerWeights(layer, eta, lambda);
        }
    }

    /**
     * Load a model from a JSON file
     *
     * @param file A JSON file
     * @throws IOException
     */
    @Override
    public void load(File file) throws IOException
    {
        load(new FileInputStream(file));
    }

    /**
     * Save a model to a JSON file.  This only writes parameters necessary for prediction, not training
     *
     * @param file A JSON file
     * @throws IOException
     */
    @Override
    public void save(File file) throws IOException
    {
        save(new FileOutputStream(file));
    }

    /**
     * Load a model from a JSON stream
     *
     * @param inputStream Any input stream for JSON
     * @throws IOException
     */
    @Override
    public void load(InputStream inputStream) throws IOException
    {
        JsonNode rootNode = OBJECT_MAPPER.readTree(inputStream);
        JsonNode doScaling = rootNode.get("ScaleOutput");
        this.scaleOutput = doScaling.asBoolean();
        JsonNode subtree = rootNode.get("Layers");
        Iterator<JsonNode> children = subtree.iterator();
        List<Layer> list = new ArrayList<Layer>();

        for (; children.hasNext(); )
        {
            JsonNode child = children.next();
            Map<String, Object> params = OBJECT_MAPPER.treeToValue(child, Map.class);
            list.add(Layers.toLayer(params));
        }
        layers = list.toArray(new Layer[list.size()]);
        inputStream.close();
    }

    /**
     * Save a model to a JSON stream.  This only writes parameters necessary for prediction, not training
     *
     * @param outputStream A JSON output stream
     * @throws IOException
     */
    @Override
    public void save(OutputStream outputStream) throws IOException
    {
        ObjectNode rootNode = OBJECT_MAPPER.createObjectNode();
        List<Map<String, Object>> layerArray = new ArrayList<Map<String, Object>>(layers.length);
        for (int i = 0; i < layers.length; ++i)
        {
            layerArray.add(Layers.toParams(layers[i]));
        }

        JsonNode subtree = OBJECT_MAPPER.valueToTree(layerArray);
        rootNode.put("Layers", subtree);
        rootNode.put("ScaleOutput", scaleOutput);

        OBJECT_MAPPER.writerWithDefaultPrettyPrinter().writeValue(outputStream, rootNode);
        outputStream.close();

    }

    /**
     * Override the SGDTk base model's fit() function to predict.  This gives back a binary prediction centered around
     * 0.  Values that are positive are 'true', values that are negative are 'false'.  Note that for non-binary cases,
     * you should use NeuralNetModel.score(), not predict().  In the case of predict() for softmax output, it will
     * simply give you the (log) probability under the best class (but wont tell you which -- not very useful
     * outside the library, so use score()!!)
     *
     * @param fv A feature vector
     * @return A value centered at zero, where, in the case of classification, negative values indicate 'false'
     * and positive indicate 'true'
     */
    @Override
    public double predict(FeatureVector fv)
    {
        double[] scores = score(fv);
        double mx = scores[0];
        for (int i = 1; i < scores.length; ++i)
        {
            mx = Math.max(scores[i], mx);
        }
        return mx;
    }

    /**
     * Forward prop
     * @param x A feature vector
     * @return A result
     */
    private Tensor forward(Tensor x)
    {

        Tensor z = x;
        for (int i = 0; i < layers.length; ++i)
        {

            Layer layer = layers[i];

            z = layer.forward(z);

        }

        return z;
    }

    /**
     * Give back the the output layer of the network, scaling if required
     *
     * @param fv Feature vector
     * @return Return an array of scores.  In the simple binary case, we get a single value back.  In the case of a
     * log softmax type output, we get an array of values, where the index into the array is (label - 1).  The highest
     * score is then the prediction
     */
    @Override
    public double[] score(FeatureVector fv)
    {

        DenseVectorN dvn = (DenseVectorN)fv.getX();
        ArrayDouble x = dvn.getX();
        Tensor tensor = new Tensor(x, x.size());
        Tensor output = forward(tensor);

        int sz = output.size();
        double[] scores = new double[sz];


        System.arraycopy(output.getArray().v, 0, scores, 0, sz);

        // Assuming a probability distribution, we are going to want to shift and scale
        if (scaleOutput)
        {
            for (int i = 0; i < sz; ++i)
            {
                scores[i] = 2 * (scores[i] - 0.5);
            }

        }
        return scores;
    }

    /**
     * Not doing this right now, so dont make us select the learning rate in SGDTk unless you desire abject failure.
     * @return Nothing!
     */
    @Override
    public Model prototype()
    {
        return null;
    }
}
