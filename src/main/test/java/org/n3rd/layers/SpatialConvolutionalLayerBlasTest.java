package org.n3rd.layers;

import com.beust.jcommander.Parameter;
import org.junit.Test;
import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;

import static junit.framework.TestCase.assertEquals;

public class SpatialConvolutionalLayerBlasTest
{
    double[] OF_2x2 = {
            14, 20, 15, 24,
            12, 24, 17, 26
    };

    double[] OIK_2_3_2x2 = {
            1, 1, 2, 2,
            1, 1, 1, 1,
            0, 1, 1, 0,
            1, 0, 0, 1,
            2, 1, 2, 1,
            1, 2, 2, 0
    };
    double[] IF_3_3x3 = {
            1, 2, 0,
            1, 1, 3,
            0, 2, 2,
            0, 2, 1,
            0, 3, 2,
            1, 1, 0,
            1, 2, 1,
            0, 1, 3,
            3, 3, 2};


    double[] IFM_COL_MAJ = {
            1, 2, 1, 1,
            2, 0, 1, 3,
            1, 1, 0, 2,
            1, 3, 2, 2,
            0, 2, 0, 3,
            2, 1, 3, 2,
            0, 3, 1, 1,
            3, 2, 1, 0,
            1, 2, 0, 1,
            2, 1, 1, 3,
            0, 1, 3, 3,
            1, 3, 3, 2
    };

    double[] KM_COL_MAJ = {
            1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0,
            1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0
    };

    double[] OFM = {14, 20, 15, 24, 12, 24, 17, 26 };


    private void unwrapInput(Tensor unwrappedInput, Tensor x,  int kL, int kH, int kW)
    {


        int z = 0;

        assertEquals(x.dims[1], kL);
        final int iH = x.dims[1];
        final int iW = x.dims[2];

        final int oH = iH - kH + 1;
        final int oW = iW - kW + 1;

        for (int k = 0; k < kL; ++k)
        {

            for (int m = 0; m < kH; ++m)
            {
                for (int n = 0; n < kW; ++n)
                {
                    // Cycle all taps at each kernel?
                    for (int i = 0; i < oH; ++i)
                    {
                        for (int j = 0; j < oW; ++j)
                        {
                            // This is then image(k, i + m, j + n)
                            int offset = (k * iH + i + m) * iW + j + n;
                            unwrappedInput.set(z, x.at(offset));
                            ++z;
                        }
                    }
                }
            }
        }

    }
    @Test
    public void testUnwrapStrategy() throws Exception
    {
        final int rows = 12;
        final int cols = 4;
        Tensor x = new Tensor(IF_3_3x3, 3, 3, 3);
        Tensor unwrapped = new Tensor(rows, cols);
        unwrapInput(unwrapped, x, 3, 2, 2);
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                double d = unwrapped.get(i * cols + j);

                assertEquals(d, IFM_COL_MAJ[i * cols + j], 1e-6);
            }

        }
    }

    @Test
    public void testForward() throws Exception
    {
        Tensor x = new Tensor(IF_3_3x3, 3, 3, 3);
        SpatialConvolutionalLayerBlas layer = new SpatialConvolutionalLayerBlas(2, 2, 2, x.dims);

        for (int i = 0; i < layer.getParams().size(); ++i)
        {
            layer.getParams().set(i, KM_COL_MAJ[i]);
        }

        Tensor z = layer.forward(x);

        for (int i = 0; i < z.size(); ++i)
        {
            assertEquals(z.get(i), OFM[i]);
        }



    }
    @Test
    public void testBackward() throws Exception
    {
        Tensor x = new Tensor(IF_3_3x3, 3, 3, 3);
        SpatialConvolutionalLayerBlas layer = new SpatialConvolutionalLayerBlas(2, 2, 2, x.dims);

        for (int i = 0; i < layer.getParams().size(); ++i)
        {
            layer.getParams().set(i, KM_COL_MAJ[i]);
        }

        Tensor z = layer.forward(x);

        SpatialConvolutionalLayer layer2 = new SpatialConvolutionalLayer(2, 2, 2, x.dims);
        for (int i = 0; i < layer2.getParams().size(); ++i)
        {
            layer2.getParams().set(i, OIK_2_3_2x2[i]);
        }
        Tensor z2 = layer2.forward(x);
        for (int i = 0; i < z2.size(); ++i)
        {
            assertEquals(z.get(i), z2.get(i));
        }

        Tensor chainG = new Tensor(z);
        chainG.scale(0.0001);

        Tensor e = layer.backward(chainG, 1);

        Tensor e2 = layer2.backward(chainG, 1);

        for (int i = 0; i < e2.size(); ++i)
        {
            assertEquals(e.get(i), e2.get(i), 1e-6);
        }


    }

}