
package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;

/**
 * Fully connected layer
 *
 * Implements a fully connected layer.  Pretty self-explanatory
 *
 * @author dpressel
 */
public class FullyConnectedLayer extends AbstractLayer
{

    protected Tensor z;
    private int outputLength;
    private int inputLength;

    /**
     * This method exists for deserialization from the model
     *
     * @param params The params, which are immediately converted to a Tensor object
     */
    public void setParams(LinkedHashMap<String, Object> params)
    {
        this.weights = new Tensor(params);
    }

    public void setBiasParams(ArrayList<Double> biasParams)
    {
        int sz = biasParams.size();
        biases = new double[sz];
        for (int i = 0; i < sz; ++i)
        {
            biases[i] = biasParams.get(i);
        }
    }

    /**
     * Random initializer for weights
     *
     * @return
     */
    public double rand()
    {
        double stdv = 1. / Math.sqrt(inputLength);
        double stdv2 = stdv * 2;
        return Math.random() * stdv2 - stdv;
    }

    /**
     * Empty constructor (for reincarnating models)
     */
    public FullyConnectedLayer()
    {

    }

    /**
     * Constructor, with given outputLength and input length
     * @param outputLength Output length
     * @param inputLength Input length
     */
    public FullyConnectedLayer(int outputLength, int inputLength)
    {
        this.outputLength = outputLength;
        this.inputLength = inputLength;
        weights = new Tensor(outputLength, this.inputLength);
        gradsW = new Tensor(outputLength, this.inputLength);
        output = new Tensor(outputLength);
        biases = new double[outputLength];
        biasGrads = new double[outputLength];

        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += this.inputLength)
        {
            for (int j = 0; j < this.inputLength; ++j)
            {
                weights.set(ibase + j, rand());
            }
            biases[i] = rand();
        }
        grads = new Tensor(this.inputLength);
        //z = new Tensor(inputLength);
    }

    /**
     * Forward prop
     * @param x
     * @return
     */
    @Override
    public Tensor forward(Tensor x)
    {
        //x.copyTo(z);
        z = x;
        return fX(z, weights);
    }

    protected Tensor fX(Tensor x, Tensor w)
    {

        final int zL = Math.min(inputLength, x.size());
        ArrayDouble oA = output.getArray();

        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
        {
            double acc = 0.;
            for (int j = 0; j < zL; ++j)
            {
                acc += w.at(ibase + j) * x.at(j);
            }

            oA.set(i, acc + biases[i]);
        }
        return output;
    }

    /**
     * Do backprop
     * @param chainGrad layers above's deltas
     * @param y Label
     * @return The deltas for this layer
     */
    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        final int zLength = z.size();
        final int howLong = Math.min(inputLength, zLength);

        grads.constant(0.);
        //gradsW.constant(0.);

        ArrayDouble gwA = gradsW.getArray();
        ArrayDouble gA = grads.getArray();
        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
        {
            
            // They should be the same, but make sure
            // Calculate gradients WRT weights
            // Because of how we are doing our outputs, we know we do not have sparse layers


            final double cgi = chainGrad.at(i);
            for (int j = 0; j < howLong; ++j)
            {
                gwA.addi(ibase + j, cgi * z.at(j));
                gA.addi(j, cgi * weights.at(ibase + j));
            }
            // push propagates through on a constant term
            biasGrads[i] += cgi;
        }

        //// GRADIENT CHECKING.  We have x laying around, so we need to do the forward computation on both
        ////gradCheck(outputLayerGradArray);
        ////gradCheckX(outputLayerGradArray);
        return grads;

    }

    // dx on a fully connected layer is simply the parameters by the parent gradient
    void gradCheckX(Tensor outputLayerGradArray)
    {

        double sumX = 0.0;

        for (int i = 0; i < grads.size(); ++i)
        {
            sumX += grads.at(i);
        }

        double sumNumGrad = 0.;
        for (int i = 0; i < inputLength; ++i)
        {
            double xd = z.at(i);
            double xdp = xd + 1e-4;
            double xdm = xd - 1e-4;
            // first add it
            z.set(i, xdp);
            Tensor zop = fX(z, weights);
            z.set(i, xdm);
            Tensor zom = fX(z, weights);
            z.set(i,  xd);

            for (int j = 0; j < zop.size(); ++j)
            {
                double dxx = (zop.at(j) - zom.at(j)) / (2*1e-4);
                dxx *= outputLayerGradArray.at(j);
                sumNumGrad += dxx;
            }
        }
        double absDelta = Math.abs(sumNumGrad - sumX);
        if (absDelta > 1e-6)
        {
            System.out.println("Abs delta large: " + absDelta);
        }
    }

    // When we shift parameters, isolating each, we get the gradient WRT the parameters
    void gradCheck(Tensor outputLayerGradArray)
    {

        double sumX = 0.0;

        for (int i = 0, sz = gradsW.size(); i < sz; ++i)
        {
            sumX += gradsW.at(i);
        }
        double sumNumGrad = 0.;

        for (int i = 0, sz = weights.size(); i < sz; ++i)
        {
            double wd = weights.at(i);
            double wdp = wd + 1e-4;
            double wdm = wd - 1e-4;
            // first add it
            weights.set(i, wdp);

            Tensor zop = fX(z, weights);
            weights.set(i, wdm);
            Tensor zom = fX(z, weights);
            weights.set(i, wd);
            for (int j = 0, zsz = zop.size(); j < zsz; ++j)
            {
                double dxx = outputLayerGradArray.at(j) * (zop.at(j) - zom.at(j))/ (2*1e-4);
                sumNumGrad += dxx;
            }
        }

        double absDelta = Math.abs(sumNumGrad - sumX);
        if (absDelta > 1e-6)
        {
            System.out.println("Abs delta large: " + absDelta);
        }
    }

    public int getOutputLength()
    {
        return outputLength;
    }

    public void setOutputLength(Integer outputLength)
    {
        this.outputLength = outputLength;
    }

    public int getInputLength()
    {
        return inputLength;
    }

    public void setInputLength(Integer inputLength)
    {
        this.inputLength = inputLength;
    }
}
