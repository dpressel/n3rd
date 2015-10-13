package org.n3rd.layers;

import org.n3rd.Tensor;

/**
 * Folding layer, inspired by Kalchbrenner & Blunsom's folding layer, but more general purpose
 *
 * This layer folds word vector dims down.  Unlike the K&B version, it does an average which was more intuitive to me
 * and generalizes to multifolds better, and we support more than just pairwise folding -- you specify the
 * number of folds with the optional third (k) parameter
 * This class (obviously), does no learning itself
 *
 * @author dpressel
 */
public class AverageFoldingLayer extends AbstractLayer
{

    private int embeddingSz;
    private int featureMapSz;
    private int numFrames;
    private int k;
    public AverageFoldingLayer()
    {

    }
    public AverageFoldingLayer(int featureMapSz, int embedSz)
    {
        this(featureMapSz, embedSz, 2);
    }

    public AverageFoldingLayer(int featureMapSz, int embedSz, int k)
    {

        this.embeddingSz = embedSz;
        this.featureMapSz = featureMapSz;
        this.k = k;
    }
    @Override
    public Tensor forward(Tensor z)
    {
        numFrames = z.size()/embeddingSz/featureMapSz;
        final int outEmbeddingSz = embeddingSz/k;

        // Do a resize() here!
        output = new Tensor(featureMapSz, outEmbeddingSz, numFrames);
        output.reset(0.);

        double div = 1.0 / k;
        for (int l = 0; l < featureMapSz; ++l)
        {
            for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
            {
                for (int i = 0; i < numFrames; ++i)
                {
                    int oAddr = (l * outEmbeddingSz + p) * numFrames + i;


                    output.d[oAddr] = 0.0;
                    for (int m = 0; m < k; ++m)
                    {
                        int iAddr = (l * embeddingSz + j + m) * numFrames + i;
                        output.d[oAddr] += z.d[iAddr];
                    }
                    output.d[oAddr] *= div;
                }
            }
        }
        return output;
    }

    // Since the output and input are the same for the max value, we can just apply the
    // max-pool value from the output
    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {

        grads = new Tensor(featureMapSz, embeddingSz, numFrames);
        double div = 1.0 / k;
        int outEmbeddingSz = embeddingSz/k;
        for (int l = 0; l < featureMapSz; ++l)
        {
            for (int i = 0; i < numFrames; ++i)
            {
                for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
                {
                    int oAddr = (l * outEmbeddingSz + p) * numFrames + i;
                    double value = chainGrad.d[oAddr] * div;
                    for (int m = 0; m < k; ++m)
                    {
                        int iAddr = (l * embeddingSz + j + m) * numFrames + i;
                        grads.d[iAddr] = value;
                    }
                }
            }
        }

        return grads;
    }

    public int getEmbeddingSz()
    {
        return embeddingSz;
    }

    public void setEmbeddingSz(Integer embeddingSz)
    {
        this.embeddingSz = embeddingSz;
    }

    public int getFeatureMapSz()
    {
        return featureMapSz;
    }

    public void setFeatureMapSz(Integer featureMapSz)
    {
        this.featureMapSz = featureMapSz;
    }

}
