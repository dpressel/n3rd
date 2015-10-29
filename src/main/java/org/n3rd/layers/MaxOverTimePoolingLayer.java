package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;
import org.sgdtk.Offset;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Max over time pooling will take the max pixel from each feature map, and spit out a result.
 * So, if you have featureMapSize 300 for example, and you do  max over time pooling on any N-length
 * signal, you are going to get only 300 outputs.  This type of pooling is very fast, but doesnt preserve
 *
 *
 * @author dpressel
 */
public class MaxOverTimePoolingLayer extends AbstractLayer
{
    private int featureMapSz;
    int numFrames;
    int[] origin;

    /**
     * Default constructor, used prior to rehydrating model from file
     */
    public MaxOverTimePoolingLayer()
    {

    }

    /**
     * Constructor for training
     *
     * @param featureMapSz This is the number of feature maps
     */
    public MaxOverTimePoolingLayer(int featureMapSz)
    {
        this.featureMapSz = featureMapSz;
        output = new Tensor(featureMapSz);
        origin = new int[output.size()];
        grads = new Tensor(1);
    }

    public int getFeatureMapSz()
    {
        return featureMapSz;
    }

    public void setFeatureMapSz(Integer featureMapSz)
    {
        this.featureMapSz = featureMapSz;
    }

    @Override
    public Tensor forward(Tensor z)
    {


        numFrames = z.size() / featureMapSz;
        grads.resize(featureMapSz, 1, numFrames);
        int sz = output.size();

        ArrayDouble oA = output.getArray();
        final ArrayDouble zA = z.getArray();
        for (int i = 0; i < sz; ++i)
        {
            oA.set(i, 0);
            origin[i] = -100;
        }

        for (int l = 0; l < featureMapSz; ++l)
        {

            int mxIndex = 0;
            double mxValue = -100;

            for (int i = 0; i < numFrames; ++i)
            {

                int inAddr = l * numFrames + i;
                double ati = zA.at(inAddr);
                if (ati > mxValue)
                {
                    mxIndex = inAddr;
                    mxValue = ati;
                }

            }

            origin[l] = mxIndex;
            oA.set(l, mxValue);

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

        for (int l = 0; l < featureMapSz; ++l)
        {

            int inAddr = origin[l];
            gA.set(inAddr, chainGrad.at(l));

        }
        return grads;
    }

}
