package org.n3rd.layers;

import org.n3rd.Tensor;
import org.n3rd.ops.FilterOps;
import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

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
public class TemporalConvolutionalLayer implements Layer
{
    // we have an output number of feature maps, an input width and an input number of feature maps
    // e.g. 500 x 5 x 300
    Tensor weights;

    // Input is Number of frames x frame width (num feature maps)
    Tensor input;

    // Output is Number of frames x num feature maps


    Tensor gradsW;
    Tensor grads;
    double[] biases;
    double[] biasGrads;

    double forwardTime = 0.;
    int currentForward = 0;
    double backwardTime = 0.;
    int currentBackward = 0;

    public TemporalConvolutionalLayer()
    {

    }

    public TemporalConvolutionalLayer(int nK, int kL, int kW, int embeddingSize)
    {

        weights = new Tensor(nK, kL, kW, embeddingSize);
        gradsW = new Tensor(nK, kL, kW, embeddingSize);
        biases = new double[nK];
        biasGrads = new double[nK];


        // For each kernel, randomly initialize all weights
        for (int i = 0; i < weights.d.length; ++i)
        {
            weights.d[i] = rand();
        }

    }

    public double rand()
    {

        //final int embeddingSz = weights.dims[3];
        double stdv = 1. / Math.sqrt(6. / 28.);
        double stdv2 = stdv * 2;
        double d = Math.random() * stdv2 - stdv;
        return d;
        //return 1;
    }

    @Override
    public VectorN forward(VectorN z)
    {
        // For convolutions, we should assume that our VectorN is truly a matrix
        // and the usual math applies
        DenseVectorN dv = (DenseVectorN) z;
        final int inputFeatureMapSz = weights.dims[1];
        final int embeddingSz = weights.dims[3];
        final int numFrames = z.length() / embeddingSz / inputFeatureMapSz;
        double[] mx = dv.getX();
        input = new Tensor(mx, inputFeatureMapSz, numFrames, embeddingSz);
        grads = new Tensor(inputFeatureMapSz, numFrames, embeddingSz);
        long start = System.currentTimeMillis();
        // 10x faster or more than using corr1MM
        Tensor output = FilterOps.corr1(input, weights, biases); // biases
        long elapsed = System.currentTimeMillis() - start;
        this.forwardTime += elapsed;
        this.currentForward++;
        if (currentForward % 100000 == 0)
        {
            System.out.println("Fwd (ms/vec): " + this.forwardTime / currentForward);
        }
        return new DenseVectorN(output.d);

    }

    @Override
    public VectorN backward(VectorN chainGrad, double y)
    {
        final int featureMapSz = weights.dims[0];
        final int embeddingSz = weights.dims[3];
        final int kW = weights.dims[2];
        final int numFrames = input.dims[1];
        final int convOutputSz = numFrames - kW + 1;
        // The desired dims going backwards is going to be

        int[] outputDims = new int[] { featureMapSz, convOutputSz, embeddingSz };
        DenseVectorN chainGradDense = (DenseVectorN) chainGrad;

        double[] chainGradX = chainGradDense.getX();
        Tensor chainGradTensor = new Tensor(chainGradX, outputDims);
        int stride = convOutputSz * embeddingSz;
        for (int l = 0; l < featureMapSz; ++l)
        {
            for (int i = 0; i < stride; ++i)
            {
                this.biasGrads[l] += chainGradX[l * stride + i];
            }
            this.biasGrads[l] /= embeddingSz;
        }

        int zpFrameSize = numFrames + kW - 1;
        int zp = zpFrameSize - convOutputSz;

        Tensor zpChainGrad = Tensor.embed(chainGradTensor, zp, 0);
        Tensor tWeights = Tensor.transposeWeight4D(weights);
        long start = System.currentTimeMillis();

        Tensor gradUps = FilterOps.conv1(zpChainGrad, tWeights, null);
        long elapsed = System.currentTimeMillis() - start;
        this.backwardTime += elapsed;
        this.currentBackward++;

        for (int i = 0; i < gradUps.d.length; ++i)
        {
            this.grads.d[i] += gradUps.d[i];
        }

        Tensor gradWUps = FilterOps.corr1Weights(input, chainGradTensor);

        // Try not moving the gradient of weights
        for (int i = 0; i < weights.d.length; ++i)
        {
            gradsW.d[i] += gradWUps.d[i];
        }

        //// GRADIENT CHECK
        ////gradCheck(chainGradTensor);
        ////gradCheckX(chainGradTensor);
        return new DenseVectorN(this.grads.d);
    }

    void gradCheckX(Tensor outputLayerGradArray)
    {

        double sumX = 0.;
        for (int i = 0; i < grads.d.length; ++i)
        {
            sumX += grads.d[i];
        }

        double sumNumGrad = 0.;
        for (int i = 0; i < input.d.length; ++i)
        {
            double xd = input.d[i];
            double xdp = xd + 1e-4;
            double xdm = xd - 1e-4;

            input.d[i] = xdp;
            Tensor outputHigh = FilterOps.corr1(input, weights, biases);

            input.d[i] = xdm;
            Tensor outputLow = FilterOps.corr1(input, weights, biases);

            input.d[i] = xd;
            for (int j = 0; j < outputHigh.d.length; ++j)
            {
                double dxx = (outputHigh.d[j] - outputLow.d[j]) / (2 * 1e-4);
                sumNumGrad += outputLayerGradArray.d[j] * dxx;
            }
        }
        double absDelta = Math.abs(sumNumGrad - sumX);
        if (absDelta > 1e-6)
        {
            System.out.println("Abs delta large: " + absDelta);
        }

    }

    // When we shift parameters, isolating each, we get the gradient WRT the parameters
    // In order to find the mathematical gradient, look at the
    void gradCheck(Tensor chainGradTensor)
    {

        double sumNumGrad = 0.;
        double sumGrad = 0.0;
        for (int i = 0; i < gradsW.d.length; ++i)
        {
            sumGrad += this.gradsW.d[i];
        }
        // Each weight sees every x except xi < w.length

        //weights = new Tensor(nK, kL, kW, embeddingSize);
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
                        double wd = weights.d[z];
                        double wdp = wd + 1e-4;
                        double wdm = wd - 1e-4;
                        // first add it
                        weights.d[z] = wdp;
                        Tensor outputHigh = FilterOps.corr1(input, weights, biases);

                        weights.d[z] = wdm;
                        Tensor outputLow = FilterOps.corr1(input, weights, biases);

                        weights.d[z] = wd;
                        ++z;

                        for (int w = 0; w < outputHigh.d.length; ++w)
                        {
                            double dxx = chainGradTensor.d[w] * (outputHigh.d[w] - outputLow.d[w]) / (2 * 1e-4);
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
