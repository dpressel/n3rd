package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;
import org.sgdtk.Offset;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * K-max pooling, a generalization of max-pooling over time, where we take the top K values
 * <p>
 * K-max pooling layer implements temporal max pooling, selecting up to K max features.  This is the approach used
 * in Kalchbrenner & Blunsom for their CNN sentence classification.  When K is 1, it simply becomes max-pooling over
 * time.
 * <p>
 * The current implementation just uses builtin Java data structures, and isnt likely to be particularly optimal
 * and can likely be simplified quite a bit.
 *
 * @author dpressel
 */
public class KMaxPoolingLayer extends AbstractLayer
{

    private int k;
    private int embeddingSz;
    private int featureMapSz;
    int numFrames;
    int[] origin;


    /**
     * Default constructor, used prior to rehydrating model from file
     */
    public KMaxPoolingLayer()
    {

    }

    /**
     * Constructor for training
     *
     * @param k            The number of max values to use in each embedding
     * @param featureMapSz This is the number of feature maps
     * @param embedSz      This is the embedding space, e.g, for some word2vec input, this might be something like 300
     */
    public KMaxPoolingLayer(int k, int featureMapSz, int embedSz)
    {
        this.k = k;
        this.embeddingSz = embedSz;
        this.featureMapSz = featureMapSz;
        output = new Tensor(featureMapSz, embeddingSz, k);
        origin = new int[output.size()];
        grads = new Tensor(1);
    }

    public int getK()
    {
        return k;
    }

    public void setK(Integer k)
    {
        this.k = k;
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

    public static final class MaxValueComparator implements Comparator<Offset>
    {
        @Override
        public int compare(Offset o1, Offset o2)
        {
            //return o1.index;
            return Double.compare(o2.value, o1.value);
        }
    }

    public static final class MinIndexComparator implements Comparator<Offset>
    {
        @Override
        public int compare(Offset o1, Offset o2)
        {
            //return o1.index;
            return Integer.compare(o1.index, o2.index);
        }
    }

    @Override
    public Tensor forward(Tensor z)
    {


        numFrames = z.size() / embeddingSz / featureMapSz;
        grads.resize(featureMapSz, embeddingSz, numFrames);
        //grads = new Tensor(featureMapSz, embeddingSz, numFrames);
        int sz = output.size();

        ArrayDouble oA = output.getArray();
        final ArrayDouble zA = z.getArray();
        for (int i = 0; i < sz; ++i)
        {
            oA.set(i, 0);
            origin[i] = -100;
        }

        for (int l = 0, lbase = 0; l < featureMapSz; ++l, lbase += embeddingSz)
        {
            for (int j = 0; j < embeddingSz; ++j)
            {
                List<Offset> offsets = new ArrayList<Offset>(numFrames);

                final int ibase = (lbase + j) * numFrames;
                final int obase = (lbase + j) * k;

                for (int i = 0; i < numFrames; ++i)
                {
                    int inAddr = ibase + i;
                    offsets.add(new Offset(inAddr, zA.at(inAddr)));

                }
                offsets.sort(new MaxValueComparator());
                List<Offset> offsetList = offsets.subList(0, Math.min(k, offsets.size()));
                offsetList.sort(new MinIndexComparator());
                sz = offsetList.size();
                for (int i = 0; i < sz; ++i)
                {
                    int outAddr = obase + i;
                    origin[outAddr] = offsetList.get(i).index;
                    oA.set(outAddr, offsetList.get(i).value);
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

        grads.constant(0.);
        ArrayDouble gA = grads.getArray();

        for (int l = 0, lbase = 0; l < featureMapSz; ++l, lbase += embeddingSz)
        {

            for (int j = 0; j < embeddingSz; ++j)
            {
                int obase = (lbase + j) * k;
                for (int i = 0; i < k; ++i)
                {
                    int outAddr = obase + i;
                    int inAddr = origin[outAddr];
                    if (inAddr == -100)
                    {
                        continue;
                    }
                    gA.set(inAddr, chainGrad.at(outAddr));
                }
            }
        }
        return grads;
    }

}
