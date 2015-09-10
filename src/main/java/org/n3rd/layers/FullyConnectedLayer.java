
package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

import java.util.ArrayList;
import java.util.LinkedHashMap;

/**
 * Fully connected layer
 *
 * Implements a fully connected layer.  Pretty self-explanatory
 *
 * @author dpressel
 */
public class FullyConnectedLayer implements Layer
{

    protected DenseVectorN z;
    protected Tensor weights;
    protected Tensor gradW;
    protected double[] biases;
    protected double[] gradBiases;
    private int outputLength;
    private int inputLength;
    protected DenseVectorN dx;

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
        gradW = new Tensor(outputLength, this.inputLength);
        biases = new double[outputLength];
        gradBiases = new double[outputLength];
        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += this.inputLength)
        {
            for (int j = 0; j < this.inputLength; ++j)
            {
                weights.d[ibase + j] = rand();
            }
            biases[i] = rand();
        }
        dx = new DenseVectorN(this.inputLength);

    }

    /**
     * Forward prop
     * @param x
     * @return
     */
    @Override
    public VectorN forward(VectorN x)
    {
        this.z = new DenseVectorN(x);
        return fX(z.getX(), weights.d);
    }

    // This is the workhorse
    protected DenseVectorN fX(double[] x, double[] w)
    {

        DenseVectorN preAct = new DenseVectorN(outputLength);
        double[] preActV = preAct.getX();

        int zL = Math.min(inputLength, x.length);
        //System.out.println("zL " + zL + ", " + outputLength + " x " + inputLength);
        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
        {
            double acc = 0.;
            for (int j = 0; j < zL; ++j)
            {
                acc += w[ibase + j] * x[j];
            }

            preActV[i] = acc + biases[i];
        }
        return preAct;
    }


    @Override
    public Tensor getParamGrads()
    {
        return gradW;
    }

    @Override
    public Tensor getParams()
    {
        return weights;
    }

    @Override
    public double[] getBiasGrads()
    {
        return gradBiases;
    }

    @Override
    public double[] getBiasParams()
    {
        return biases;
    }

    /**
     * Do backprop
     * @param outputLayerGrad layers above's deltas
     * @param y Label
     * @return The deltas for this layer
     */
    @Override
    public VectorN backward(VectorN outputLayerGrad, double y)
    {

        dx.reset();
        DenseVectorN outputLayerGradDense = (DenseVectorN)outputLayerGrad;
        double [] dxV = dx.getX(); 
        double [] outputLayerGradArray = outputLayerGradDense.getX();
        int zLength = z.length();
        int howLong = Math.min(inputLength, zLength);
        double [] zArray = z.getX();

        for (int i = 0, ibase = 0; i < outputLength; ++i, ibase += inputLength)
        {
            
            // They should be the same, but make sure
            // Calculate gradients WRT weights
            // Because of how we are doing our outputs, we know we do not have sparse layers


            for (int j = 0; j < howLong; ++j)
            {

                gradW.d[ibase + j] += outputLayerGradArray[i] * zArray[j];
                dxV[j] += outputLayerGradArray[i] * weights.d[ibase + j];
                
            }
            // push propagates through on a constant term
            gradBiases[i] += outputLayerGradArray[i];
        }

        //// GRADIENT CHECKING.  We have x laying around, so we need to do the forward computation on both
        ////gradCheck(outputLayerGradArray);
        ////gradCheckX(outputLayerGradArray);
        return dx;

    }

    // dx on a fully connected layer is simply the parameters by the parent gradient
    void gradCheckX(double[] outputLayerGradArray)
    {
        double[] dxArray = dx.getX();
        double sumX = 0.0;

        for (int i = 0; i < dxArray.length; ++i)
        {
            sumX += dxArray[i];
        }

        double[] zArray = z.getX();
        double sumNumGrad = 0.;
        for (int i = 0; i < inputLength; ++i)
        {
            double xd = zArray[i];
            double xdp = xd + 1e-4;
            double xdm = xd - 1e-4;
            // first add it
            zArray[i] = xdp;
            double[] zop = fX(zArray, weights.d).getX();
            zArray[i] = xdm;
            double[] zom = fX(zArray, weights.d).getX();
            zArray[i] = xd;

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
        double[] zArray = z.getX();
        double sumX = 0.0;
        //for (int j = 0; j < zArray.length; ++j)
        for (int i = 0; i < gradW.d.length; ++i)
        {
            sumX += gradW.d[i];
        }
        double sumNumGrad = 0.;
        for (int i = 0; i < weights.d.length; ++i)
        {
            double wd = weights.d[i];
            double wdp = wd + 1e-4;
            double wdm = wd - 1e-4;
            // first add it
            weights.d[i] = wdp;
            double[] x = z.getX();
            double[] zop = fX(x, weights.d).getX();
            weights.d[i] = wdm;
            double[] zom = fX(x, weights.d).getX();
            weights.d[i] = wd;
            for (int j = 0; j < zop.length; ++j)
            {
                double dxx = outputLayerGradArray[j] * (zop[j] - zom[j])/ (2*1e-4);
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
