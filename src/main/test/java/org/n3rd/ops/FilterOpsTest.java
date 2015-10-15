package org.n3rd.ops;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;

// At this point, I have tested convolution and correlation in 2D and 1D, and zp embed in 1D.
// The next thing to check is the backprop version
// for the weights, which is a bit tricky
public class FilterOpsTest
{

    double[] D =
            {
                    1,3,5,7,9,11,
                    2,4,6,8,10,12
            };

    double[] K = {
            1,3,5,
            2,4,6
    };

    double[] O_NARROW = {
            35,53,71,89,
            56,80,104,128
    };
    double[] O_CONV_NARROW = {
            19,37,55,73,
            40,64,88,112 };
    double[] O2_NARROW = { 91.0, 133.0, 175.0, 217.0 };

    double[] O2_CONV_NARROW = { 56, 98, 140, 182 };

    double[] O_WIDE = {
            5,18,35,53,71,89,42,11,
            12,32,56,80,104,128,68,24
         };
    double[] O_CONV_WIDE = {
            1, 6, 19, 37, 55, 73, 78, 55,
            4, 16, 40, 64, 88, 112, 108, 72
    };

    @Test
    public void testCorr1() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        Tensor kernel = new Tensor(K, new int[]{1,1,2,3});
        Tensor output = new Tensor(1, 2, O_NARROW.length/2);
        FilterOps.corr1(data, kernel, null, output);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], 2);
        assertEquals(output.dims[2], O_NARROW.length/2);
        for (int i = 0; i < O_NARROW.length; ++i)
        {
            assertEquals(output.get(i), O_NARROW[i]);
        }

    }

    @Test
    public void testConv2() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        Tensor kernel = new Tensor(K, new int[]{1,1,2,3});
        Tensor output = new Tensor(1, 2, O2_CONV_NARROW.length/2);
        FilterOps.conv2(data, kernel, null, output);
        assertEquals(output.dims[0], 1);
        for (int i = 0; i < O2_CONV_NARROW.length; ++i)
        {
            assertEquals(output.get(i), O2_CONV_NARROW[i]);
        }
    }

    @Test
    public void testCorr2() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        Tensor kernel = new Tensor(K, new int[]{1,1,2,3});
        Tensor output = new Tensor(1, 2, O2_NARROW.length/2);
        FilterOps.corr2(data, kernel, null, output);
        assertEquals(output.dims[0], 1);
        for (int i = 0; i < O2_NARROW.length; ++i)
        {
            assertEquals(output.get(i), O2_NARROW[i]);
        }
    }

    @Test
    public void testCorr2MultiFM() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        double[] k = new double[2*K.length];
        for (int i = 0; i < K.length; ++i)
        {
            k[i] = k[K.length+i] = K[i];
        }
        Tensor kernel = new Tensor(k, new int[]{2,1,2,3});
        Tensor output = new Tensor(2, 1, 4);
                FilterOps.corr2(data, kernel, null, output);
        assertEquals(output.dims[0], 2);
        for (int i = 0; i < O2_NARROW.length; ++i)
        {
            double o2n = O2_NARROW[i];
            assertEquals(output.get(i), o2n);
            assertEquals(output.get(O2_NARROW.length+i), o2n);
        }
    }

    @Test
    public void testCorr1MultiFM() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        double[] k = new double[2*K.length];
        for (int i = 0; i < K.length; ++i)
        {
            k[i] = k[K.length+i] = K[i];
        }
        Tensor kernel = new Tensor(k, new int[]{2,1,2,3});
        Tensor output = new Tensor(2, 2, 4);
        FilterOps.corr1(data, kernel, null, output);
        assertEquals(output.dims[0], 2);
        for (int i = 0; i < O_NARROW.length; ++i)
        {
            double on = O_NARROW[i];
            assertEquals(output.get(i), on);
            assertEquals(output.get(O_NARROW.length+i), on);
        }
    }



    @Test
    public void testWideCorr1() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        Tensor zp = data.embed(0, 4);
        Tensor kernel = new Tensor(K, new int[]{1,1,2,3});

        Tensor output = new Tensor(1, 2, 8);
        FilterOps.corr1(zp, kernel, null, output);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], 2);
        assertEquals(output.dims[2], O_WIDE.length/2);
        for (int i = 0; i < O_WIDE.length; ++i)
        {
            assertEquals(output.get(i), O_WIDE[i]);
        }

    }
    @Test
    public void testWideConv1() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        Tensor zp = data.embed(0, 4);
        Tensor kernel = new Tensor(K, new int[]{1,1,2,3});

        Tensor output = new Tensor(1, 2, 8);
        FilterOps.conv1(zp, kernel, null, output);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], 2);
        assertEquals(output.dims[2], O_CONV_WIDE.length/2);
        for (int i = 0; i < O_CONV_WIDE.length; ++i)
        {
            assertEquals(output.get(i), O_CONV_WIDE[i]);
        }

    }

    @Test
    public void testConv1() throws Exception
    {
        Tensor data = new Tensor(D, new int[]{1, 2, 6});
        Tensor kernel = new Tensor(K, new int[]{1, 1,2,3});
        Tensor output = new Tensor(1, 2, 4);
        FilterOps.conv1(data, kernel, null, output);
        assertEquals(output.dims[0], 1);
        assertEquals(output.dims[1], 2);
        assertEquals(output.dims[2], O_CONV_NARROW.length/2);
        for (int i = 0; i < O_CONV_NARROW.length; ++i)
        {
            assertEquals(output.get(i), O_CONV_NARROW[i]);
        }
    }

}