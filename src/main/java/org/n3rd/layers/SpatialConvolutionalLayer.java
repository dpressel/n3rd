package org.n3rd.layers;

import org.n3rd.Tensor;
import org.n3rd.ops.FilterOps;
import org.sgdtk.ArrayDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;

/**
 * "Spatial" or "2D" convolution
 *
 * 2D convolution typically is performed in neural nets over multi-channel images or feature maps, using a kernel which
 * is a 4D tensor (when we are not doing batch SGD) of number of output feature maps, number of input feature maps,
 * kernel height, and kernel width -- referred to here as (nK, kL, kH, kW)
 *
 * @author dpressel
 */
public class SpatialConvolutionalLayer extends AbstractLayer
{
    // Cube represents multiple feature maps for this layer
    Tensor input;

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

    int[] inputDims;

    public SpatialConvolutionalLayer()
    {

    }
    public SpatialConvolutionalLayer(int nK, int kH, int kW, int... inputDims)
    {

        final int iL = inputDims.length == 3? inputDims[0]: 1;
        final int iH = inputDims[1];
        final int iW = inputDims[2];

        weights = new Tensor(nK, iL, kH, kW);
        gradsW = new Tensor(nK, iL, kH, kW);
        biases = new double[nK];
        biasGrads = new double[nK];

        this.grads = new Tensor(inputDims);

        // For each kernel, randomly initialize all weights
        output  = new Tensor( nK, iH - kH + 1, iW - kW + 1);

        for (int i = 0, sz = weights.size(); i < sz; ++i)
        {
            weights.set(i, rand());
        }

    }

    public double rand()
    {
        //double stdv = 1. / Math.sqrt(input.dims[1] * input.dims[2]);
        double stdv = 1. / Math.sqrt(grads.dims[1] * weights.dims[2] * weights.dims[3]);
        double stdv2 = stdv * 2;
        return Math.random() * stdv2 - stdv;
    }

    @Override
    public Tensor forward(Tensor z)
    {
        input = new Tensor(z.getArray(), grads.dims);
        //z.copyTo(input);
        FilterOps.corr2(input, weights, biases, output);
        return output;
    }

    // For every filter, do a convolution
    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {
        //final int iL = input.dims[0];
        final int iH = input.dims[1];
        final int iW = input.dims[2];

        final int oH = output.dims[1];
        final int oW = output.dims[2];
        //final int kL = weights.dims[1];
        final int kH = weights.dims[2];
        final int kW = weights.dims[3];

        final int zpH = iH + kH - 1;
        final int zpW = iW + kW - 1;

        Tensor zpChainGradCube = chainGrad.embed(zpH - oH, zpW - oW);

        int nK = weights.dims[0];

        Tensor tWeights = weights.transposeWeight4D();

        // This is actually what is failing.  Why?  Probably a bug in transpose weight 4D?
        FilterOps.conv2(zpChainGradCube, tWeights, null, grads);

        // This is correct, we know that the gradient of the weights is checking out
        FilterOps.corr2Weights(input, chainGrad, gradsW);

        for (int l = 0; l < nK; ++l)
        {
            for (int i = 0, sz = chainGrad.size(); i < sz; ++i)
            {
                this.biasGrads[l] += chainGrad.at(i);
            }
        }

        //// GRADIENT CHECK
        ////gradCheck(chainGradTensor);
        ////gradCheckX(chainGradTensor);

        return grads;
    }


    void gradCheckX(ArrayDouble outputLayerGradArray)
    {

        double sumX = 0.;

        for (int i = 0, sz = grads.size(); i < sz; ++i)
        {
            sumX += grads.at(i);
        }

        double sumNumGrad = 0.;

        for (int i = 0, sz = input.size(); i < sz; ++i)
        {
            double xd = input.at(i);
            double xdp = xd + 1e-4;
            double xdm = xd - 1e-4;
            input.set(i, xdp);
            Tensor outputHigh = new Tensor(output.dims);
            Tensor outputLow = new Tensor(output.dims);
            FilterOps.corr2(input, weights, biases, outputHigh);
            input.set(i, xdm);
            FilterOps.corr2(input, weights, biases, outputLow);
            input.set(i, xd);

            for (int j = 0, zsz = outputHigh.size(); j < zsz; ++j)
            {
                double dxx = (outputHigh.at(j) - outputLow.at(j)) / (2 * 1e-4);
                sumNumGrad += outputLayerGradArray.at(j) * dxx;
            }
        }
        double absDelta = Math.abs(sumNumGrad - sumX);
        if (absDelta > 1e-6)
        {
            System.out.println("X abs delta large: " + absDelta);
        }

    }

    // When we shift parameters, isolating each, we get the gradient WRT the parameters
    // In order to find the mathematical gradient, look at the
    void gradCheck(ArrayDouble chainGrad)
    {

        double sumNumGrad = 0.;
        double sumGrad = 0.0;

        for (int i = 0; i < gradsW.size(); ++i)
        {
            sumGrad += gradsW.at(i);
        }

        int z = 0;
        for (int k = 0; k < weights.dims[0]; ++k)
        {
            for (int l = 0; l < weights.dims[1]; ++l)
            {
                // kW
                for (int i = 0; i < weights.dims[2]; ++i)
                {
                    for (int j = 0; j < weights.dims[3]; ++j)
                    {
                        double wd = weights.at(z);
                        double wdp = wd + 1e-4;
                        double wdm = wd - 1e-4;
                        // first add it
                        weights.set(z, wdp);

                        Tensor outputHigh = new Tensor(output.dims);
                        FilterOps.corr2(input, weights, biases, outputHigh);

                        weights.set(z, wdm);

                        Tensor outputLow = new Tensor(output.dims);
                        FilterOps.corr2(input, weights, biases, outputLow);

                        weights.set(z, wd);

                        ++z;

                        for (int w = 0; w < outputHigh.size(); ++w)
                        {
                            double dxx = chainGrad.at(w) * (outputHigh.at(w) - outputLow.at(w)) / (2 * 1e-4);
                            sumNumGrad += dxx;
                        }
                    }
                }
            }
        }
        double absDelta = Math.abs(sumNumGrad - sumGrad);
        if (absDelta > 1e-6)
        {
            System.out.println("Abs delta large: " + absDelta);
        }
    }

}
