package org.n3rd.layers;

import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

/**
 * Spatial max pooling layer on a fixed height/width input
 *
 * This is the typical max pooling for image processing.  It will not work for temporal or variable length
 * input
 *
 * @author dpressel
 */
public class MaxPoolingLayer extends AbstractLayer
{

    private int[] inputDims;
    private int[] origin;

    // Could save some memory here, but since size is fixed, we are reusing it over and over this way

    int dh;
    int dw;
    public MaxPoolingLayer(int dh, int dw, int... inputDims)
    {

        this.inputDims = new int[inputDims.length];
        for (int i = 0; i < inputDims.length; ++i)
        {
            this.inputDims[i] = inputDims[i];
        }
        this.grads = new Tensor(inputDims);

        this.dh = dh;
        this.dw = dw;

        this.output = new Tensor(inputDims[0],
                (int)Math.ceil(inputDims[1] / (double)dh),
                (int)Math.ceil(inputDims[2] / (double)dw));

        this.origin = new int[output.size()];
    }
    @Override
    public Tensor forward(Tensor z)
    {


        ArrayDouble oA = output.getArray();

        for (int i = 0; i < origin.length; ++i)
        {
            oA.set(i, -100);
            origin[i] = 0;
        }
        final int kL = inputDims[0];
        final int iH = inputDims[1];
        final int iW = inputDims[2];
        final int oH = output.dims[1];
        final int oW = output.dims[2];
        for (int l = 0; l < kL; ++l)
        {
            for (int i = 0; i < iH; ++i)
            {
                int oi = (int) Math.floor(i / (double) dh);

                for (int j = 0; j < iW; ++j)
                {
                    int oj = (int) Math.floor(j / (double) dw);
                    int outAddr = (l * oH + oi) * oW + oj;
                    int inAddr = (l * iH + i) * iW + j;

                    final double zi = z.at(inAddr);

                    if (oA.at(outAddr) < zi)
                    {
                        oA.set(outAddr, zi);
                        origin[outAddr] = inAddr;
                    }
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
        final int kL = inputDims[0];
        final int iH = inputDims[1];
        final int iW = inputDims[2];
        final int oH = output.dims[1];
        final int oW = output.dims[2];
        ArrayDouble gA = grads.getArray();

        for (int l = 0; l < kL; ++l)
        {
            for (int i = 0; i < iH; ++i)
            {
                int oi = (int)Math.floor(i / (double) dh);

                for (int j = 0; j < iW; ++j)
                {
                    int oj = (int)Math.floor(j / (double) dw);
                    int outAddr = (l *oH + oi) * oW + oj;
                    int inAddr = (l * iH + i) * iW + j;
                    gA.set(inAddr, origin[outAddr] == inAddr ? chainGrad.at(outAddr) : 0.);
                }
            }
        }
        return grads;
    }

    // We have no params in this layer
    @Override
    public Tensor getParamGrads()
    {
        return null;
    }

    @Override
    public Tensor getParams()
    {
        return null;
    }

    @Override
    public double[] getBiasGrads()
    {
        return null;
    }

    @Override
    public double[] getBiasParams()
    {
        return null;
    }

    public int[] getInputDims()
    {
        return inputDims;
    }

    public void setInputDims(Integer[] inputDims)
    {
        this.inputDims = new int[inputDims.length];
        for (int i = 0; i < inputDims.length; ++i)
        {
            this.inputDims[i] = inputDims[i];
        }
    }

}
