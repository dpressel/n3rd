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
        output = new Tensor(featureMapSz, numFrames, outEmbeddingSz);
        output.reset(0.);

        double div = 1.0 / k;
        for (int l = 0; l < featureMapSz; ++l)
        {
            for (int i = 0; i < numFrames; ++i)
            {
                int obase = (l * numFrames + i) * outEmbeddingSz;
                int ibase = (l * numFrames + i) * embeddingSz;
                for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
                {
                    output.d[obase + p] = 0.0;
                    for (int m = 0; m < k; ++m)
                    {
                        output.d[obase + p] += z.d[ibase + j + m];
                    }
                    output.d[obase + p] *= div;
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

        grads = new Tensor(featureMapSz, numFrames, embeddingSz);
        double div = 1.0 / k;
        int outEmbeddingSz = embeddingSz/k;
        for (int l = 0; l < featureMapSz; ++l)
        {
            for (int i = 0; i < numFrames; ++i)
            {
                int obase = (l * numFrames + i) * outEmbeddingSz;
                int ibase = (l * numFrames + i) * embeddingSz;
                for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
                {
                    double value = chainGrad.d[obase + p] * div;
                    for (int m = 0; m < k; ++m)
                    {
                        grads.d[ibase + j + m] = value;
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
