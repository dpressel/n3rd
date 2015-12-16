package org.n3rd.layers;

import org.n3rd.Tensor;
import org.n3rd.ops.FilterOps;

import java.util.ArrayList;
import java.util.LinkedHashMap;

/**
 * Temporal convolution with support for feature maps AND preserving word embeddings
 * <p>
 * This thing is different from the one in Torch.  In torch, the frame width is essentially a feature map
 * and the output is also.  This means that embeddings are not preserved between layers.
 * Assuming we want to preserve that locally, we would do this differently, making the embedding size 1,
 * and using the nK for the embeddingSz.  I believe this basically means that we can do everything Torch can, but also
 * we can do the Kalchbrenner/Blunsom thing as well. If you want to do the Torch approach, just pass and embeddingSz
 * of 1 and handle everything else outside
 */
public class TemporalConvolutionalLayer extends AbstractLayer
{

    // we have an output number of feature maps, an input width and an input number of feature maps

    // Input is Number of frames x frame width (num feature maps)
    Tensor input;

    public TemporalConvolutionalLayer()
    {

    }

    public TemporalConvolutionalLayer(int nK, int kL, int kW)
    {
        this(nK, kL, kW, 1);
    }

    public TemporalConvolutionalLayer(int nK, int kL, int kW, int embeddingSize)
    {

        weights = new Tensor(nK, kL, embeddingSize, kW);
        gradsW = new Tensor(nK, kL, embeddingSize, kW);
        biases = new double[nK];
        biasGrads = new double[nK];

        for (int i = 0, sz = weights.size(); i < sz; ++i)
        {
            weights.set(i, rand());
        }
        output = new Tensor(1);
        grads = new Tensor(1);

    }

    public double rand()
    {

        //final int embeddingSz = weights.dims[3];
        double stdv = 1. / Math.sqrt(6. / 28.);
        double stdv2 = stdv * 2;
        //double d = RND[Current++ % RND.length] * stdv2 - stdv;
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
            grads.resize(kL, embeddingSz, numFrames);
            //grads = new Tensor(kL, embeddingSz, numFrames);
            output.resize(nK, embeddingSz, oT);
            //output = new Tensor(nK, embeddingSz, oT);
            FilterOps.corr1(input, weights, biases, output);
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

            chainGrad.reshape(featureMapSz, embeddingSz, convOutputSz);
            // The desired dims going backwards is going to be
            int stride = convOutputSz * embeddingSz;
            for (int l = 0; l < featureMapSz; ++l)
            {
                //this.biasGrads[l] = 0;
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
            FilterOps.conv1(zpChainGrad, tWeights, null, grads);
            FilterOps.corr1Weights(input, chainGrad, gradsW);

            //// GRADIENT CHECK
            ////gradCheck(chainGradTensor);
            ////gradCheckX(chainGradTensor);
            return grads;
        }
        catch (Exception ex)
        {
            throw new RuntimeException(ex);
        }
    }

    void gradCheckX(Tensor outputLayerGrad)
    {

        double sumX = 0.;
        for (int i = 0, sz = grads.size(); i < sz; ++i)
        {
            sumX += grads.at(i);
        }

        double sumNumGrad = 0.;
        for (int i = 0, sz = input.size(); i < sz; ++i)
        {
            double xd =  input.at(i);
            double xdp = xd + 1e-4;
            double xdm = xd - 1e-4;

            input.set(i, xdp);
            Tensor outputHigh = new Tensor(output.dims);
            Tensor outputLow = new Tensor(output.dims);

            FilterOps.corr1(input, weights, biases, outputHigh);

            input.set(i, xdm);
            FilterOps.corr1(input, weights, biases, outputLow);

            input.set(i, xd);


            for (int j = 0, zsz = outputHigh.size(); j < zsz; ++j)
            {
                double dxx = (outputHigh.at(j) - outputLow.at(j)) / (2 * 1e-4);
                sumNumGrad += outputLayerGrad.at(j) * dxx;
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
    void gradCheck(Tensor chainGrad)
    {

        double sumNumGrad = 0.;
        double sumGrad = 0.0;

        for (int i = 0, sz = gradsW.size(); i < sz; ++i)
        {
            sumGrad += gradsW.at(i);
        }
        // Each weight sees every x except xi < w.length

        //weights = new Tensor(nK, kL, kW, embeddingSize);
        int nK = weights.dims[0];
        int kL = weights.dims[1];
        int embedSz = weights.dims[2];
        int kW = weights.dims[3];

        int z = 0;

        for (int k = 0; k < nK; ++k)
        {
            for (int l = 0; l < kL; ++l)
            {
                for (int j = 0; j < embedSz; ++j)
                {
                    // kW
                    for (int i = 0; i < kW; ++i)
                    {

                        double wd = weights.at(z);
                        double wdp = wd + 1e-4;
                        double wdm = wd - 1e-4;
                        // first add it
                        weights.set(z, wdp);
                        Tensor outputHigh = new Tensor(output.dims);
                        Tensor outputLow = new Tensor(output.dims);

                        FilterOps.corr1(input, weights, biases, outputHigh);

                        weights.set(z, wdm);

                        FilterOps.corr1(input, weights, biases, outputLow);

                        weights.set(z, wd);

                        ++z;

                        for (int w = 0, zsz = outputHigh.size(); w < zsz; ++w)
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
