package org.n3rd.ops;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;

// At this point, I have tested convolution and correlation in 2D and 1D, and zp embed in 1D.
// The next thing to check is the backprop version
// for the weights, which is a bit tricky
public class FilterOpsTest
{

    // embeddings are
    double[] D =
            {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10,
                    11, 12
            };
    double[] K = { 1, 2,
            3, 4,
            5, 6 };


    double[] KT = { 1, 3, 5, 2, 4, 6 };

    double[] KT2 = { 1, 3, 5, 2, 4, 6, 1, 3, 5, 2, 4, 6 };

    double[] O_NARROW = { 35.0, 56.0, 53.0, 80.0, 71.0, 104.0, 89.0, 128.0 };
    double[] O_CONV_NARROW = { 19.0, 40.0, 37.0, 64.0, 55.0, 88.0, 73.0, 112.0 };
    double[] O2_NARROW = { 91.0, 133.0, 175.0, 217.0 };
    double[] O2_NARROW2 = { 91.0, 133.0, 175.0, 217.0 };
    double[] O2_CONV_NARROW = { 56, 98, 140, 182 };

    double[] O_WIDE = { 5.0, 12.0, 18.0, 32.0, 35.0, 56.0, 53.0, 80.0, 71.0, 104.0, 89.0, 128.0, 42.0, 68.0, 11.0, 24.0 };
    double[] O_CONV_WIDE = { 1, 4, 6, 16, 19, 40, 37, 64, 55, 88, 73, 112, 78, 108, 55, 72 };

    double[] transpose(double[] x, int dH, int dW)
    {
        double[] d = new double[x.length];
        for (int i = 0; i < dH; ++i)
        {
            for (int j = 0; j < dW; ++j)
            {
                d[j*dH + i] = x[i*dW+j];
            }
        }
        return d;
    }
    // 1 x kW/sL x Es
    @Test
    public void testCorr1() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        Tensor kernel = new Tensor(K, new int[]{1,1,3,2});
        Tensor output = FilterOps.corr1(data, kernel, null);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], O_NARROW.length/2);
        assertEquals(output.dims[2], 2);
        for (int i = 0; i < O_NARROW.length; ++i)
        {
            assertEquals(output.d[i], O_NARROW[i]);
        }

    }

    // In Torch, with a TemporalConvolution, we do not have an option to preserve embedding depth
    // this means that each
    @Test
    public void testTorchStyleEmbeddingsCorr1() throws Exception
    {
        Tensor data = new Tensor(transpose(D, 6, 2), new int[]{2, 6, 1});
        Tensor kernel = new Tensor(KT2, new int[]{2,2,3,1});
        Tensor output = FilterOps.corr1(data, kernel, null);
        assertEquals(output.dims[0], 2);
        for (int i = 0; i < O2_NARROW2.length; ++i)
        {
            assertEquals(output.d[i], O2_NARROW2[i]);
        }


    }


    @Test
    public void testConv2() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        Tensor kernel = new Tensor(K, new int[]{1,1,3,2});
        Tensor output = FilterOps.conv2(data, kernel, null);
        assertEquals(output.dims[0], 1);
        for (int i = 0; i < O2_CONV_NARROW.length; ++i)
        {
            assertEquals(output.d[i], O2_CONV_NARROW[i]);
        }
    }

    @Test
    public void testCorr2() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        Tensor kernel = new Tensor(K, new int[]{1,1,3,2});
        Tensor output = FilterOps.corr2(data, kernel, null);
        assertEquals(output.dims[0], 1);
        for (int i = 0; i < O2_NARROW.length; ++i)
        {
            assertEquals(output.d[i], O2_NARROW[i]);
        }
    }

    @Test
    public void testCorr2MultiFM() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        double[] k = new double[2*K.length];
        for (int i = 0; i < K.length; ++i)
        {
            k[i] = k[K.length+i] = K[i];
        }
        Tensor kernel = new Tensor(k, new int[]{2,1,3,2});
        Tensor output = FilterOps.corr2(data, kernel, null);
        assertEquals(output.dims[0], 2);
        for (int i = 0; i < O2_NARROW.length; ++i)
        {
            double o2n = O2_NARROW[i];
            assertEquals(output.d[i], o2n);
            assertEquals(output.d[O2_NARROW.length+i], o2n);
        }
    }

    @Test
    public void testCorr1MultiFM() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        double[] k = new double[2*K.length];
        for (int i = 0; i < K.length; ++i)
        {
            k[i] = k[K.length+i] = K[i];
        }
        Tensor kernel = new Tensor(k, new int[]{2,1,3,2});
        Tensor output = FilterOps.corr1(data, kernel, null);
        assertEquals(output.dims[0], 2);
        for (int i = 0; i < O_NARROW.length; ++i)
        {
            double on = O_NARROW[i];
            assertEquals(output.d[i], on);
            assertEquals(output.d[O_NARROW.length+i], on);
        }
    }



    @Test
    public void testWideCorr1() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        Tensor zp = Tensor.embed(data, 4, 0);
        Tensor kernel = new Tensor(K, new int[]{1,1,3,2});

        Tensor output = FilterOps.corr1(zp, kernel, null);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], O_WIDE.length/2);
        assertEquals(output.dims[2], 2);
        for (int i = 0; i < O_WIDE.length; ++i)
        {
            assertEquals(output.d[i], O_WIDE[i]);
        }

    }

    @Test
    public void testConv1() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        Tensor kernel = new Tensor(K, new int[]{1, 1,3,2});
        Tensor output = FilterOps.conv1(data, kernel, null);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], O_CONV_NARROW.length/2);
        assertEquals(output.dims[2], 2);
        for (int i = 0; i < O_CONV_NARROW.length; ++i)
        {
            assertEquals(output.d[i], O_CONV_NARROW[i]);
        }
    }

    // Now test with multiple output feature maps
    @Test
    public void testWideConv1() throws Exception
    {
        // Narrow
        // x x x x x x
        // 0 y y y y 0
        // ow = x0W - kW + 1
        // Wide
        // 0 x x x x 0
        // 0 y y y y 0
        // x0W + kW - 1
        Tensor data = new Tensor(D, new int[]{1, 6, 2});
        Tensor zp = Tensor.embed(data, 4, 0);
        Tensor kernel = new Tensor(K, new int[]{1,1,3,2});

        Tensor output = FilterOps.conv1(zp, kernel, null);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], O_CONV_WIDE.length/2);
        assertEquals(output.dims[2], 2);
        for (int i = 0; i < O_CONV_WIDE.length; ++i)
        {
            assertEquals(output.d[i], O_CONV_WIDE[i]);
        }

    }
}