package org.n3rd.layers;

import org.n3rd.Tensor;
import org.n3rd.ops.FFTOps;
import org.n3rd.ops.FilterOps;

import java.util.ArrayList;
import java.util.LinkedHashMap;

/**
 * Temporal convolution with support for feature maps AND preserving word embeddings
 * <p/>
 * This thing is different from the one in Torch.  In torch, the frame width is essentially a feature map
 * and the output is also.  This means that embeddings are not preserved between layers.
 * Assuming we want to preserve that locally, we would do this differently, making the embedding size 1,
 * and using the nK for the embeddingSz.  I believe this basically means that we can do everything Torch can, but also
 * we can do the Kalchbrenner/Blunsom thing as well. If you want to do the Torch approach, just pass and embeddingSz
 * of 1 and handle everything else outside
 */
public class TemporalConvolutionalLayerFFT extends AbstractLayer
{

    // we have an output number of feature maps, an input width and an input number of feature maps

    // Input is Number of frames x frame width (num feature maps)
    Tensor input;

    FFTOps fft = new FFTOps();
    public TemporalConvolutionalLayerFFT()
    {

    }

    public TemporalConvolutionalLayerFFT(int nK, int kL, int kW, int embeddingSize)
    {

        weights = new Tensor(nK, kL, embeddingSize, kW);
        gradsW = new Tensor(nK, kL, embeddingSize, kW);
        biases = new double[nK];
        biasGrads = new double[nK];

        // For each kernel, randomly initialize all weights
        for (int i = 0, sz = weights.size(); i < sz; ++i)
        {
            weights.set(i, rand());
        }
        output = new Tensor(1);
        grads = new Tensor(1);

    }

    public double rand()
    {
        double stdv = 1. / Math.sqrt(6. / 28.);
        double stdv2 = stdv * 2;
        double d = Math.random() * stdv2 - stdv;
        return d;
    }

    @Override
    public Tensor forward(Tensor z)
    {
        // For convolutions, we should assume that our VectorN is truly a matrix
        // and the usual math applies

        final int nK = weights.dims[0];
        final int kL = weights.dims[1];
        final int embeddingSz = weights.dims[2];
        final int kW = weights.dims[3];

        final int numFrames = z.size() / embeddingSz / kL;
        final int oT = numFrames - kW + 1;
        try
        {
            input = new Tensor(z.getArray(), kL, embeddingSz, numFrames);
            //grads.resize(kL, embeddingSz, numFrames);
            grads = new Tensor(kL, embeddingSz, numFrames);
            //output.resize(nK, embeddingSz, oT);
            output = new Tensor(nK, embeddingSz, oT);
            FilterOps.fftfilt(fft, input, weights, biases, true, output);
            return output;
        }
        catch (Exception ex)
        {
            throw new RuntimeException(ex);
        }

    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {

        try
        {
            final int featureMapSz = weights.dims[0];
            final int embeddingSz = weights.dims[2];
            final int kW = weights.dims[3];
            final int numFrames = input.dims[2];
            final int convOutputSz = numFrames - kW + 1;

            // This is done so that the embed doesnt fail
            chainGrad.reshape(featureMapSz, embeddingSz, convOutputSz);
            //grads.constant(0.);
            int stride = convOutputSz * embeddingSz;
            for (int l = 0; l < featureMapSz; ++l)
            {
                for (int i = 0; i < stride; ++i)
                {
                    this.biasGrads[l] += chainGrad.at(l * stride + i);
                }
                this.biasGrads[l] /= embeddingSz;
            }

            int zpFrameSize = numFrames + kW - 1;
            int zp = zpFrameSize - convOutputSz;

            Tensor zpChainGrad = chainGrad.embed(0, zp);
            Tensor tWeights = weights.transposeWeight4D();

            grads.constant(0.);
            gradsW.constant(0.);
            FilterOps.fftfilt(fft, zpChainGrad, tWeights, null, false, grads);
            FilterOps.corr1Weights(input, chainGrad, gradsW);

            // No gradient check, see standard impl. for a working version
            // This class is compared against the standard impl. for unit tests
            return grads;
        }
        catch (Exception ex)
        {
            throw new RuntimeException(ex);
        }
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

}
