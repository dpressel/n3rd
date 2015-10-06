
package org.n3rd.layers;

import org.n3rd.Tensor;

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
        //return RND[Current++ % RND.length] * stdv2 - stdv;
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
                weights.d[ibase + j] = rand();
            }
            biases[i] = rand();
        }
        grads = new Tensor(this.inputLength);

    }

    /**
     * Forward prop
     * @param x
     * @return
     */
    @Override
    public Tensor forward(Tensor x)
    {
        this.z = new Tensor(x);
        return fX(z, weights);
    }

    // This is the workhorse
    protected Tensor fX(Tensor x, Tensor w)
    {



        int zL = Math.min(inputLength, x.size());
        //System.out.println("zL " + zL + ", " + outputLength + " x " + inputLength);
        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
        {
            double acc = 0.;
            for (int j = 0; j < zL; ++j)
            {
                acc += w.d[ibase + j] * x.d[j];
            }

            output.d[i] = acc + biases[i];
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
        int zLength = z.size();
        int howLong = Math.min(inputLength, zLength);

        grads.reset(0.);

        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
        {
            
            // They should be the same, but make sure
            // Calculate gradients WRT weights
            // Because of how we are doing our outputs, we know we do not have sparse layers


            for (int j = 0; j < howLong; ++j)
            {

                gradsW.d[ibase + j] += chainGrad.d[i] * z.d[j];
                grads.d[j] += chainGrad.d[i] * weights.d[ibase + j];
                
            }
            // push propagates through on a constant term
            biasGrads[i] += chainGrad.d[i];
        }

        //// GRADIENT CHECKING.  We have x laying around, so we need to do the forward computation on both
        ////gradCheck(outputLayerGradArray);
        ////gradCheckX(outputLayerGradArray);
        return grads;

    }

    // dx on a fully connected layer is simply the parameters by the parent gradient
    void gradCheckX(double[] outputLayerGradArray)
    {
        double[] dxArray = grads.d;
        double sumX = 0.0;

        for (int i = 0; i < dxArray.length; ++i)
        {
            sumX += dxArray[i];
        }

        double sumNumGrad = 0.;
        for (int i = 0; i < inputLength; ++i)
        {
            double xd = z.d[i];
            double xdp = xd + 1e-4;
            double xdm = xd - 1e-4;
            // first add it
            z.d[i] = xdp;
            double[] zop = fX(z, weights).d;
            z.d[i] = xdm;
            double[] zom = fX(z, weights).d;
            z.d[i] = xd;

            for (int j = 0; j < zop.length; ++j)
            {
                double dxx = (zop[j] - zom[j]) / (2*1e-4);
                dxx *= outputLayerGradArray[j];
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
    void gradCheck(double[] outputLayerGradArray)
    {

        double sumX = 0.0;
        //for (int j = 0; j < zArray.length; ++j)
        for (int i = 0, sz = gradsW.size(); i < sz; ++i)
        {
            sumX += gradsW.d[i];
        }
        double sumNumGrad = 0.;
        for (int i = 0, sz = weights.size(); i < sz; ++i)
        {
            double wd = weights.d[i];
            double wdp = wd + 1e-4;
            double wdm = wd - 1e-4;
            // first add it
            weights.d[i] = wdp;

            Tensor zop = fX(z, weights);
            weights.d[i] = wdm;
            Tensor zom = fX(z, weights);
            weights.d[i] = wd;
            for (int j = 0, zsz = zop.size(); j < zsz; ++j)
            {
                double dxx = outputLayerGradArray[j] * (zop.d[j] - zom.d[j])/ (2*1e-4);
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
