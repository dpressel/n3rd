package org.n3rd.layers;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;

public class SpatialConvolutionalLayerTest
{

    // embeddings are
    double[] D =
            {
                    1,2,
                    3,4,
                    5,6,
                    7,8,
                    9,10,
                    11,12
            };
    double[] D2 =
            {
                    1,2,
                    3,4,
                    5,6,
                    7,8,
                    9,10,
                    11,12,
                    2,4,
                    6,8,
                    10,12,
                    14,16,
                    18,20,
                    22,24
            };
    double[] K = {1,2,
            3,4,
            5,6};
    double[] K2 = {1,2,
        3,4,
        5,6,
        2,4,
        6,8,
        10,12};

    double[] O2_NARROW = { 91.0, 133.0, 175.0, 217.0 };
    double[] O2_2TO1 = {455.0,665.0,875.0,1085.0};
    double[] O2_GRAD_INPUT_2TO1 = {
            455.0,910.0,
            2030.0,3150.0,
            5145.0,7140.0,
            7035.0,9660.0,
            7630.0,9590.0,
            5425.0,6510.0,
            910.0,1820.0,
            4060.0,6300.0,
            10290.0,14280.0,
            14070.0,19320.0,
            15260.0,19180.0,
            10850.0,13020.0
};
    double[] O2_GRAD_INPUT = {91.0, 182.0,
            406.0,630.0,
            1029.0,1428.0,
            1407.0,1932.0,
            1526.0,1918.0,
            1085.0,1302.0
    };
    double[] O2_WEIGHT_GRAD = {
            14420,17500,
            20580,23660,
            26740,29820,
            28840, 35000,
            41160, 47320,
            53480, 59640,

    };
    double[] O2_GRAD_INPUT_2TO3 = {
            6370,   12740,
            28420,  44100,
            72030,   99960,
            98490,  135240,
            106820,  134260,
            75950,   91140,
            12740,   25480,
            56840 ,  88200,
            144060,  199920,
            196980,  270480,
            213640,  268520,
            151900,  182280

    };

    double[] O2_WEIGHT_GRAD_2TO3 = {
            14420,17500,
            20580,23660,
            26740,29820,

            28840,35000,
            41160,47320,
            53480,59640,

            28840,35000,
            41160,47320,
            53480,59640,

            57680,70000,
            82320,94640,
            106960,119280,


            43260,52500,
            61740,70980,
            80220,89460,


            86520,105000,
            123480,141960,
            160440,178920
    };

    @Test
    public void testForward1To1() throws Exception
    {
        SpatialConvolutionalLayer l = new SpatialConvolutionalLayer(1, 3, 2, 1, 6, 2);
        for (int i = 0; i < K.length; ++i)
        {
            l.weights.d[i] = K[i];
        }
        Tensor d = new Tensor(D, D.length);

        Tensor output = l.forward(d);

        assertEquals(output.size(), O2_NARROW.length);
        double[] x = output.d;
        for (int i = 0; i < O2_NARROW.length; ++i)
        {
            assertEquals(x[i], O2_NARROW[i]);
        }

    }

    @Test
    public void testForward2To1() throws Exception
    {
        SpatialConvolutionalLayer l = new SpatialConvolutionalLayer(1, 3, 2, 2, 6, 2);
        for (int i = 0; i < K2.length; ++i)
        {
            l.weights.d[i] = K2[i];

        }
        Tensor d = new Tensor(D2, D2.length);
        Tensor output = l.forward(d);

        assertEquals(output.size(), O2_2TO1.length);
        double[] x = output.d;
        for (int i = 0; i < O2_2TO1.length; ++i)
        {
            assertEquals(x[i], O2_2TO1[i]);
        }

    }

    void print(Tensor x, int nK, int kL)
    {
        for (int i = 0; i < x.dims[2]; ++i)
        {
            for (int j = 0; j < x.dims[3]; ++j)
            {
                System.out.println(x.d[((nK * x.dims[1] + kL) * x.dims[2] + i) * x.dims[3] + j]);
            }
        }
    }

    @Test
    public void testForward2To3() throws Exception
    {
        SpatialConvolutionalLayer l = new SpatialConvolutionalLayer(3, 3, 2, 2, 6, 2);

        for (int i = 0; i < K2.length; ++i)
        {
            l.weights.d[i] = K2[i];
            l.weights.d[K2.length + i] = 2*K2[i];
            l.weights.d[2*K2.length + i] = 3*K2[i];

        }

        Tensor d = new Tensor(D2, D2.length);
        Tensor output = l.forward(d);

        assertEquals(output.size(), O2_2TO1.length * 3);
        double[] x = output.d;
        for (int i = 0; i < O2_2TO1.length; ++i)
        {
            assertEquals(x[i], O2_2TO1[i]);
            assertEquals(x[O2_2TO1.length + i], 2*O2_2TO1[i]);
            assertEquals(x[2*O2_2TO1.length + i], 3*O2_2TO1[i]);
        }

    }

    @Test
    public void testBackward2to3() throws Exception
    {
        SpatialConvolutionalLayer l = new SpatialConvolutionalLayer(3, 3, 2, 2, 6, 2);
        for (int i = 0; i < K2.length; ++i)
        {
            l.weights.d[i] = K2[i];
            l.weights.d[K2.length + i] = 2*K2[i];
            l.weights.d[2*K2.length + i] = 3*K2[i];

        }

        Tensor d = new Tensor(D2, D2.length);
        Tensor output = l.forward(d);
        Tensor gradI = l.backward(output, 0);
        double[] gIx = gradI.d;
        for (int i = 0; i < O2_GRAD_INPUT_2TO3.length; ++i)
        {
            assertEquals(gIx[i], O2_GRAD_INPUT_2TO3[i]);
        }

        Tensor gradW = l.getParamGrads();
        System.out.println(gradW.d);

        assertEquals(gradW.size(), O2_WEIGHT_GRAD_2TO3.length);

        for (int i = 0; i < O2_WEIGHT_GRAD_2TO3.length; ++i)
        {
            assertEquals(gradW.d[i], O2_WEIGHT_GRAD_2TO3[i]);
        }




    }

    @Test
    public void testBackward2To1() throws Exception
    {
        SpatialConvolutionalLayer l = new SpatialConvolutionalLayer(1, 3, 2, 2, 6, 2);
        for (int i = 0; i < K2.length; ++i)
        {
            l.weights.d[i] = K2[i];

        }
        Tensor d = new Tensor(D2, D2.length);
        Tensor output = l.forward(d);
        Tensor gradI = l.backward(output, 0);
        double[] gIx = gradI.d;
        for (int i = 0; i < O2_GRAD_INPUT_2TO1.length; ++i)
        {
            assertEquals(gIx[i], O2_GRAD_INPUT_2TO1[i]);
        }

        Tensor gradW = l.getParamGrads();
        assertEquals(gradW.size(), O2_WEIGHT_GRAD.length);
        for (int i = 0; i < O2_WEIGHT_GRAD.length; ++i)
        {
            assertEquals(gradW.d[i], O2_WEIGHT_GRAD[i]);
        }



    }

    @Test
    public void testBackward1To1() throws Exception
    {
        SpatialConvolutionalLayer l = new SpatialConvolutionalLayer(1, 3, 2, 1, 6, 2);
        for (int i = 0; i < K.length; ++i)
        {
            l.weights.d[i] = K[i];
        }
        Tensor d = new Tensor(D, D.length);
        Tensor output = l.forward(d);
        Tensor gradI = l.backward(output, 0);
        double[] gIx = gradI.d;
        for (int i = 0; i < O2_GRAD_INPUT.length; ++i)
        {
            assertEquals(gIx[i], O2_GRAD_INPUT[i]);
        }


    }
}