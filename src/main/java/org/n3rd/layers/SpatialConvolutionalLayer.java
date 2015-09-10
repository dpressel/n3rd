package org.n3rd.layers;

import org.n3rd.Tensor;
import org.n3rd.ops.FilterOps;
import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

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
public class SpatialConvolutionalLayer implements Layer
{
    Tensor weights;

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

    int[] outputDims;

    Tensor gradsW;
    Tensor grads;
    double[] biases;
    double[] biasGrads;

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

        this.input = new Tensor(null, inputDims);
        this.grads = new Tensor(inputDims);
        this.outputDims = new int[] { nK, iH - kH + 1, iW - kW + 1 };
        // For each kernel, randomly initialize all weights
        for (int i = 0; i < weights.d.length; ++i)
        {
            weights.d[i] = rand();
        }

    }

    public double rand()
    {
        double stdv = 1. / Math.sqrt(input.dims[1] * input.dims[2]);
        double stdv2 = stdv * 2;
        return Math.random() * stdv2 - stdv;
    }

    @Override
    public VectorN forward(VectorN z)
    {
        // For convolutions, we should assume that our VectorN is truly a matrix
        // and the usual math applies
        DenseVectorN dv = (DenseVectorN) z;

        double[] mx = dv.getX();
        input.d = mx;
        Tensor output = FilterOps.corr2(input, weights, biases);
        return new DenseVectorN(output.d);

    }

    // For every filter, do a convolution
    @Override
    public VectorN backward(VectorN chainGrad, double y)
    {
        final int iL = input.dims[0];
        final int iH = input.dims[1];
        final int iW = input.dims[2];

        final int oH = outputDims[1];
        final int oW = outputDims[2];
        final int kL = weights.dims[1];
        final int kH = weights.dims[2];
        final int kW = weights.dims[3];
        DenseVectorN chainGradDense = (DenseVectorN) chainGrad;
        double[] chainGradX = chainGradDense.getX();
        Tensor chainGradTensor = new Tensor(chainGradX, outputDims);
        final int zpH = iH + kH - 1;
        final int zpW = iW + kW - 1;

        Tensor zpChainGradCube = Tensor.embed(chainGradTensor, zpH - oH, zpW - oW);

        int nK = weights.dims[0];

        Tensor tWeights = Tensor.transposeWeight4D(weights);

        Tensor gradUps = FilterOps.conv2(zpChainGradCube, tWeights, null);
        for (int i = 0; i < gradUps.d.length; ++i)
        {
            this.grads.d[i] += gradUps.d[i];
        }

        Tensor gradWUps = FilterOps.corr2Weights(input, chainGradTensor);


        for (int i = 0; i < weights.d.length; ++i)
        {
            gradsW.d[i] += gradWUps.d[i];

        }
        for (int l = 0; l < nK; ++l)
        {
            for (int i = 0; i < chainGradX.length; ++i)
            {
                this.biasGrads[l] += chainGradX[i];
            }
        }


        return new DenseVectorN(this.grads.d);
    }

    @Override
    public Tensor getParamGrads()
    {
        return gradsW;
    }

    @Override
    public Tensor getParams()
    {
        return weights;
    }

    @Override
    public double[] getBiasGrads()
    {
        return new double[0];
    }

    @Override
    public double[] getBiasParams()
    {
        return new double[0];
    }
}
