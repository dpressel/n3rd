package org.n3rd.layers;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;

public class MaxPoolingLayerTest
{

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

    double[] O2_2TO1 = {455.0,665.0,875.0,1085.0};
    double[] O2_2TO1_MX_DS_21 = {665.0,1085.0};
    double[] O2_2TO1_MX_DS_31 = {875.0,1085.0};
    double[] O2_2TO1_MX_DS32_21 = {665.0,1085.0,665.0*2,1085.0*2,665.0*3,1085.0*3};
    double[] O2_2TO1_MX_DS32_31 = {875.0,1085.0,875.0*2,1085.0*2,875.0*3,1085.0*3};
    @Test
    public void testForward2To3p2x1() throws Exception
    {

        MaxPoolingLayer l = new MaxPoolingLayer(2,1,3,4,1);


        Tensor d = new Tensor(O2_2TO1.length * 3);



        for (int i = 0; i < O2_2TO1.length; ++i)
        {
            d.set(i, O2_2TO1[i]);
            d.set(O2_2TO1.length + i, 2*O2_2TO1[i]);
            d.set(2*O2_2TO1.length + i, 3*O2_2TO1[i]);
        }
        Tensor output = l.forward(d);
        assertEquals(6, output.size());


        for (int i = 1; i < O2_2TO1_MX_DS_21.length; ++i)
        {
            assertEquals(output.get(i), O2_2TO1_MX_DS_21[i]);
            assertEquals(output.get(O2_2TO1_MX_DS_21.length + i), 2*O2_2TO1_MX_DS_21[i]);
            assertEquals(output.get(2+O2_2TO1_MX_DS_21.length +i), 3*O2_2TO1_MX_DS_21[i]);
        }


    }

    @Test
    public void testForward2To3p3x1() throws Exception
    {

        MaxPoolingLayer l = new MaxPoolingLayer(2,1,3,4,1);


        Tensor d = new Tensor(O2_2TO1.length * 3);

        for (int i = 0; i < O2_2TO1.length; ++i)
        {
            d.set(i, O2_2TO1[i]);
            d.set(O2_2TO1.length + i, 2*O2_2TO1[i]);
            d.set(2*O2_2TO1.length + i, 3*O2_2TO1[i]);
        }
        Tensor output = l.forward(d);
        assertEquals(6, output.size());

        for (int i = 1; i < O2_2TO1_MX_DS_31.length; ++i)
        {
            assertEquals(output.get(i), O2_2TO1_MX_DS_31[i]);
            assertEquals(output.get(O2_2TO1_MX_DS_31.length + i), 2*O2_2TO1_MX_DS_31[i]);
            assertEquals(output.get(2+O2_2TO1_MX_DS_31.length + i), 3*O2_2TO1_MX_DS_31[i]);
        }


    }

    @Test
    public void testBackward2to3p2x1() throws Exception
    {

        MaxPoolingLayer l = new MaxPoolingLayer(2,1,3,4,1);
        Tensor d = new Tensor(O2_2TO1.length * 3);


        for (int i = 0; i < O2_2TO1.length; ++i)
        {
            d.set(i, O2_2TO1[i]);
            d.set(O2_2TO1.length + i, 2*O2_2TO1[i]);
            d.set(2*O2_2TO1.length + i, 3*O2_2TO1[i]);
        }
        Tensor output = l.forward(d);
        Tensor grads = l.backward(output, 0);

        int n = 0;
        for (int i = 1, sz = grads.size(); i < sz; i+=2)
        {
            if (i % 2 == 0)
            {
                assertEquals(grads.get(i), 0.0);
            }
            else
            {

                assertEquals(grads.get(i), O2_2TO1_MX_DS32_21[n]);
                ++n;
            }
        }
    }
    @Test
    public void testBackward2to3p3x1() throws Exception
    {

        MaxPoolingLayer l = new MaxPoolingLayer(3,1,3,4,1);
        Tensor d = new Tensor(O2_2TO1.length * 3);


        for (int i = 0; i < O2_2TO1.length; ++i)
        {
            d.set(i, O2_2TO1[i]);
            d.set(O2_2TO1.length + i, 2*O2_2TO1[i]);
            d.set(2*O2_2TO1.length + i, 3*O2_2TO1[i]);
        }
        Tensor output = l.forward(d);
        Tensor grads = l.backward(output, 0);

        int n = 0;
        for (int i = 2, sz = grads.size(); i < sz; i += 4, n+=2)
        {
            assertEquals(grads.get(i), O2_2TO1_MX_DS32_31[n]);
            assertEquals(grads.get(i + 1), O2_2TO1_MX_DS32_31[n+1]);

        }
    }
}