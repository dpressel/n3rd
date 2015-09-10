package org.n3rd.layers;

import org.n3rd.Tensor;
import org.n3rd.util.IntCube;
import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

/**
 * Spatial max pooling layer on a fixed height/width input
 *
 * This is the typical max pooling for image processing.  It will not work for temporal or variable length
 * input
 *
 * @author dpressel
 */
public class MaxPoolingLayer implements Layer
{

    private int [] inputDims;
    private IntCube origin;

    // Could save some memory here, but since size is fixed, we are reusing it over and over this way
    private Tensor output;
    int dh;
    int dw;
    public MaxPoolingLayer(int dh, int dw, int... inputDims)
    {

        this.inputDims = new int[inputDims.length];
        for (int i = 0; i < inputDims.length; ++i)
        {
            this.inputDims[i] = inputDims[i];
        }

        this.dh = dh;
        this.dw = dw;
        this.origin = new IntCube(inputDims[0],
                (int)Math.ceil(inputDims[1] / (double)dh),
                (int)Math.ceil(inputDims[2] / (double)dw));
        this.output = new Tensor(inputDims[0],
                (int)Math.ceil(inputDims[1] / (double)dh),
                (int)Math.ceil(inputDims[2] / (double)dw));
    }
    @Override
    public VectorN forward(VectorN x)
    {

        DenseVectorN denseVectorN = (DenseVectorN)x;
        double[] xarray = denseVectorN.getX();
        origin.reset(0);
        output.reset(0.);
        final int kL = inputDims[0];
        final int iH = inputDims[1];
        final int iW = inputDims[2];
        final int oH = output.dims[1];
        final int oW = output.dims[2];
        for (int l = 0; l < kL; ++l)
        {
            for (int i = 0; i < iH; ++i)
            {
                int oi = (int)Math.floor(i / (double)dh);

                for (int j = 0; j < iW; ++j)
                {
                    int oj = (int)Math.floor(j / (double)dw);
                    int outAddr = (l * oH + oi) * oW + oj;
                    int inAddr = (l * iH + i) * iW + j;
                    if (output.d[outAddr] < xarray[inAddr])
                    {
                        output.d[outAddr] = xarray[inAddr];
                        origin.d[outAddr] = inAddr;
                    }

                }
            }

        }
        return new DenseVectorN(output.d);
    }

    // Since the output and input are the same for the max value, we can just apply the
    // max-pool value from the output
    @Override
    public VectorN backward(VectorN chainGrad, double y)
    {
        Tensor input = new Tensor(inputDims);
        final int kL = inputDims[0];
        final int iH = inputDims[1];
        final int iW = inputDims[2];
        final int oH = output.dims[1];
        final int oW = output.dims[2];

        double[] chainGradX = ((DenseVectorN)chainGrad).getX();
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
                    input.d[inAddr] = origin.d[outAddr] == inAddr ? chainGradX[outAddr] : 0.;
                }
            }
        }
        return new DenseVectorN(input.d);
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
