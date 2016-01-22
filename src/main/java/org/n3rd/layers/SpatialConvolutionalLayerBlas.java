package org.n3rd.layers;

import org.jblas.NativeBlas;
import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

import java.util.ArrayList;
import java.util.LinkedHashMap;


public class SpatialConvolutionalLayerBlas implements Layer
{

    Tensor gradsW;
    Tensor grads;
    Tensor weightAccum;
    double[] biases;
    double[] biasGrads;
    int kL;
    int nK;
    int kH;
    int kW;
    int iH;
    int iW;

    Tensor weights;

    // Input is Number of frames x frame width (num feature maps)
    Tensor unwrappedInput;
    Tensor unwrappedGradInput;
    Tensor output;

    @Override
    public Tensor getWeightAccum()
    {
        return weightAccum;
    }

    @Override
    public Tensor getOutput()
    {
        return output;
    }



    public SpatialConvolutionalLayerBlas()
    {
    }

    public SpatialConvolutionalLayerBlas(int nK, int kH, int kW, int... inputDims)
    {

        this.nK = nK;
        this.kL = inputDims.length == 3? inputDims[0]: 1;
        this.kH = kH;
        this.kW = kW;
        this.iH = inputDims.length == 3 ? inputDims[1]: inputDims[0];
        this.iW = inputDims.length == 3 ? inputDims[2]: inputDims[1];
        // For each kernel, randomly initialize all weights
        output  = new Tensor( nK, iH - kH + 1, iW - kW + 1);
        grads = new Tensor(kL, iH, iW);



        // The unwrapped input is tap-unrolled with a width that is kH * kW * nK, and a height that is the number of lags
        unwrappedInput = new Tensor(output.dims[1]*output.dims[2], kH * kW * kL);
        unwrappedGradInput = new Tensor(unwrappedInput.dims);
        weights = new Tensor(kL * kH * kW, nK);
        weightAccum = new Tensor(kL * kH * kW, nK);
        gradsW = new Tensor(kL * kH * kW, nK);
        biases = new double[nK];
        biasGrads = new double[nK];

        for (int i = 0; i < weights.size(); ++i)
        {
            weights.set(i, rand());
        }
    }

    public double rand()
    {
        double stdv = 1. / Math.sqrt(grads.dims[1] * grads.dims[2]);
        //double stdv = 1. / Math.sqrt(grads.dims[1] * weights.dims[2] * weights.dims[3]);
        double stdv2 = stdv * 2;
        return Math.random() * stdv2 - stdv;
    }

    private void unwrapInput(ArrayDouble x)
    {


        int z = 0;

        final int oH = iH - kH + 1;
        final int oW = iW - kW + 1;

        for (int k = 0; k < kL; ++k)
        {

            for (int m = 0; m < kH; ++m)
            {
                for (int n = 0; n < kW; ++n)
                {
                    // Cycle all taps at each kernel?
                    for (int i = 0; i < oH; ++i)
                    {
                        for (int j = 0; j < oW; ++j)
                        {
                            // This is then image(k, i + m, j + n)
                            int offset = (k * iH + i + m) * iW + j + n;
                            unwrappedInput.set(z, x.at(offset));
                            ++z;
                        }
                    }
                }
            }
        }

    }

    private void wrapGrad(Tensor unwrapped)
    {

        final int oH = iH - kH + 1;
        final int oW = iW - kW + 1;


        // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
        int z = 0;
        for (int k = 0; k < kL; ++k)
        {
            for (int m = 0; m < kH; ++m)
            {
                for (int n = 0; n < kW; ++n)
                {
                    for (int i = 0; i < oH; ++i)
                    {
                        for (int j = 0; j < oW; ++j)
                        {
                            int offset = (k * iH + i + m) * iW + j + n;
                            grads.addi(offset, unwrapped.get(z));
                            z++;
                        }
                    }

                }
            }
        }



    }

    @Override
    public Tensor forward(Tensor z)
    {

        grads.constant(0.);

        ArrayDouble input = z.getArray();

        unwrapInput(input);


        final int oH = iH - kH + 1;
        final int oW = iW - kW + 1;
        for (int l = 0; l < nK; ++l)
        {
            for (int i = 0; i < oH; ++i)
            {
                for (int j = 0; j < oW; ++j)
                {
                    output.set((l * oH + i) * oW + j, biases[l]);
                }
            }
        }

        NativeBlas.dgemm('N', 'N', unwrappedInput.dims[0], weights.dims[1], unwrappedInput.dims[1], 1.0,
                unwrappedInput.getArray().v, 0, unwrappedInput.dims[0],
                weights.getArray().v, 0, weights.dims[0], 1., output.getArray().v, 0, unwrappedInput.dims[0]);

        return output;

    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {

        try
        {

            final int oH = iH - kH + 1;
            final int oW = iW - kW + 1;
            int[] outputDims = new int[]{nK, 1, oH * oW};
            chainGrad.reshape(outputDims);

            int m = outputDims[2];
            int k = nK;
            int n = weights.dims[0];

            //unwrappedGradInput.constant(0.);
            NativeBlas.dgemm('N', 'T', m, n, k, 1.0, chainGrad.getArray().v, 0, m,
                    weights.getArray().v, 0, n, 0, unwrappedGradInput.getArray().v, 0, m);

            m = unwrappedInput.dims[1];
            k = unwrappedInput.dims[0];
            n = nK;

            ///gradsW.constant(0.);
            NativeBlas.dgemm('T', 'N', m, n, k, 1.0, unwrappedInput.getArray().v, 0, k,
                    chainGrad.getArray().v, 0, k, 0, gradsW.getArray().v, 0, m);

            // We need to update gradsW, which are (kL * embeddingSize) * kW x (nK * embeddingSize);


            for (int l = 0; l < nK; ++l)
            {
                for (int i = 0; i < oH; ++i)
                {
                    for (int j = 0; j < oW; ++j)
                    {
                        this.biasGrads[l] += chainGrad.get( (l * oH + i) * oW + j);
                    }
                }

            }


            wrapGrad(unwrappedGradInput);

            return grads;
        }
        catch (Exception ex)
        {
            throw new RuntimeException(ex);
        }
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

    /**
     * Exists for reserialization purposes only!  Turns instantly into a Tensor when injected
     *
     * @param params
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

    @Override
    public double[] getBiasGrads()
    {
        return biasGrads;
    }

    @Override
    public double[] getBiasParams()
    {
        return biases;
    }

}
