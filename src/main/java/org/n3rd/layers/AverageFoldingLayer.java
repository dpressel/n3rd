package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

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
        output = new Tensor(1);
        grads = new Tensor(1);
    }
    @Override
    public Tensor forward(Tensor z)
    {
        numFrames = z.size()/embeddingSz/featureMapSz;
        final int outEmbeddingSz = embeddingSz/k;

        output.resize(featureMapSz, outEmbeddingSz, numFrames);
        //output = new Tensor(featureMapSz, outEmbeddingSz, numFrames);
        grads.resize(featureMapSz, embeddingSz, numFrames);
        double div = 1.0 / k;
        ArrayDouble oA = output.getArray();


        for (int l = 0, lbase = 0, libase = 0; l < featureMapSz; ++l, lbase += outEmbeddingSz, libase += embeddingSz)
        {
            for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
            {
                int obase = (lbase + p) * numFrames;

                for (int i = 0; i < numFrames; ++i)
                {
                    int oAddr = obase + i;
                    oA.set(oAddr, 0.0);
                    for (int m = 0; m < k; ++m)
                    {
                        int iAddr = (libase + j + m) * numFrames + i;
                        oA.addi(oAddr, z.at(iAddr));
                    }
                    oA.multi(oAddr, div);
                }
            }
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {

        //grads = new Tensor(featureMapSz, embeddingSz, numFrames);
        //
        grads.constant(0.);
        ArrayDouble gA = grads.getArray();
        double div = 1.0 / k;
        int outEmbeddingSz = embeddingSz/k;
        for (int l = 0, lbase = 0, libase = 0; l < featureMapSz; ++l, lbase += outEmbeddingSz, libase += embeddingSz)
        {
            for (int j = 0, p = 0; j < embeddingSz; j += k, ++p)
            {
                int obase = (lbase + p) * numFrames;
                for (int i = 0; i < numFrames; ++i)
                {

                    int oAddr = obase + i;
                    double value = chainGrad.at(oAddr) * div;
                    for (int m = 0; m < k; ++m)
                    {
                        int iAddr = (libase + j + m) * numFrames + i;
                        gA.set(iAddr, value);
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
