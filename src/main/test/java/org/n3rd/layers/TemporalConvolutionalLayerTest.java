package org.n3rd.layers;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;

public class TemporalConvolutionalLayerTest
{

    // embeddings are
    double[] D =
            {
                    1, 3, 5, 7, 9, 11,
                    2, 4, 6, 8, 10, 12
            };

    double[] K = {
            1, 2, 3,
            4, 5, 6
    };


    double[] OFM1IFM1 = {22, 34, 46, 58, 64, 94, 124, 154};

    double[] IFM2K = {
            1, 2, 3,
            7, 8, 9,
            4, 5, 6,
            10, 11, 12
    };

    double[] OFM1IFM2 = {
            78.5, 120.5, 162.5, 204.5,
            252.5, 366.5, 480.5, 594.5
    };

    double[] IFM2D = {
            1, 3, 5, 7, 9, 11,
            2, 4, 6, 8, 10, 12,
            1.5, 3.5, 5.5, 7.5, 9.5, 11.5,
            2.5, 4.5, 6.5, 8.5, 10.5, 12.5};

    double[] IFM2OFM3K = {

            1, 2, 3,
            7, 8, 9,

            4, 5, 6,
            10, 11, 12,


            1.5, 2.5, 3.5,
            7.5, 8.5, 9.5,

            4.5, 5.5, 6.5,
            10.5, 11.5, 12.5,


            1.2, 2.2, 3.2,
            7.2, 8.2, 9.2,

            4.2, 5.2, 6.2,
            10.2, 11.2, 12.2
    };

    double[] OFM3IFM2D = {
            78.5, 120.5, 162.5, 204.5,
            252.5, 366.5, 480.5, 594.5,

            88.25, 136.25, 184.25, 232.25,
            265.25, 385.25, 505.25, 625.25,

            82.4, 126.8, 171.2, 215.6,
            257.6, 374.0, 490.4, 606.8
    };

    double[] OFM3IFM2G_1000 = {

            0.30975499999, 1.03594, 2.312955, 3.217995, 3.14416, 2.116295,
            5.611595, 14.53462, 27.119474999, 35.778915, 28.682439999, 16.872934999,

            1.057205, 2.93404, 5.764905, 7.879545, 6.65506, 4.073345,
            7.937645, 20.23792, 37.251224999, 49.064265, 38.59054, 22.352585
    };

    double SQ_M_1000 = 3886.2073516200003;
    double SQ_M_W_1000 = 11477.130271620003;


    double[] DNOE =
            {
                    1, 3, 5, 7, 9, 11
            };


    double[] KNOE = {
            1, 2, 3

    };


    double[] OFM1IFM1NOE = {22, 34, 46, 58};

    double[] IFM2KNOE = {
            1, 2, 3,
            7, 8, 9
    };

    double[] OFM1IFM2NOE = {
            122., 182., 242., 302.
    };


    double[] IFM2DNOE = {
            1, 3, 5, 7, 9, 11,
            2, 4, 6, 8, 10, 12};


    double[] IFM2OFM3KNOE = {

            -2, -1, 0,

            1, 2, 3,

            4, 5, 6,

            7, 8, 9,

            10, 11, 12,

            -13, -14, -15
    };

    double[] OFM3IFM2DNOE = {
            23., 29., 35., 41.,
            149., 227., 305., 383.,
            -69., -87., -105., -123.
    };

    double SQ_M_1000_1CHAN = 425.789216;
    double SQ_M_W_1000_1CHAN = 2383.197216;

    double[] OFM3IFM2G_1000_1CHAN = {
            -0.14,-0.057,0.31499999999,0.873,1.090999999,0.8219999,1.963,4.953,9.072,11.7359999,9.293,5.415

    };
    @Test
    public void testForwardWordVecAsInChannels() throws Exception
    {
        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(1, 1, 3, 1);
        for (int i = 0; i < KNOE.length; ++i)
        {
            l.weights.set(i, KNOE[i]);
        }
        Tensor d = new Tensor(DNOE, 1, 1, 6);
        Tensor output = l.forward(d);

        assertEquals(output.size(), OFM1IFM1NOE.length);

        for (int i = 0; i < OFM1IFM1NOE.length; ++i)
        {
            assertEquals(output.get(i), OFM1IFM1NOE[i], 1e-6);
        }

    }

    @Test
    public void testForward2to1WordVecAsInChannels() throws Exception
    {

        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(1, 2, 3, 1);

        Tensor weights = l.getParams();
        int n = 0;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                weights.set(j * 2 + i, IFM2KNOE[n++]);
            }
        }

        Tensor d = new Tensor(IFM2DNOE, 2, 1, 6);
        Tensor output = l.forward(d);

        for (int i = 0; i < OFM1IFM2NOE.length; ++i)
        {
            assertEquals(output.get(i), OFM1IFM2NOE[i], 1e-6);
        }

    }

    @Test
    public void testForward2to3WordVecAsInChannels() throws Exception
    {

        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(3, 2, 3, 1);

        Tensor weights = l.getParams();
        assertEquals(weights.size(), IFM2OFM3KNOE.length);
        for (int i = 0; i < IFM2OFM3KNOE.length; ++i)
        {
            weights.set(i, IFM2OFM3KNOE[i]);
        }

        Tensor d = new Tensor(IFM2DNOE, 2, 1, 6);
        Tensor output = l.forward(d);

        assertEquals(output.size(), OFM3IFM2DNOE.length);
        for (int i = 0; i < OFM3IFM2DNOE.length; ++i)
        {
            assertEquals(output.get(i), OFM3IFM2DNOE[i], 1e-6);
        }

    }

    @Test
    public void testForward() throws Exception
    {
        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(1, 1, 3, 2);
        for (int i = 0; i < K.length; ++i)
        {
            l.weights.set(i, K[i]);
        }
        Tensor d = new Tensor(D, 1, 2, 6);
        Tensor output = l.forward(d);

        assertEquals(output.size(), OFM1IFM1.length);

        for (int i = 0; i < OFM1IFM1.length; ++i)
        {
            assertEquals(output.get(i), OFM1IFM1[i]);
        }

    }

    @Test
    public void testForward2to1() throws Exception
    {

        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(1, 2, 3, 2);

        Tensor weights = l.getParams();
        for (int i = 0; i < IFM2K.length; ++i)
        {
            weights.set(i, IFM2K[i]);
        }


        Tensor d = new Tensor(IFM2D, 2, 2, 6);
        Tensor output = l.forward(d);

        for (int i = 0; i < OFM1IFM2.length; ++i)
        {
            assertEquals(output.get(i), OFM1IFM2[i]);
        }

    }

    @Test
    public void testForward2to3() throws Exception
    {

        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(3, 2, 3, 2);

        Tensor weights = l.getParams();
        for (int i = 0; i < IFM2OFM3K.length; ++i)
        {
            weights.set(i, IFM2OFM3K[i]);
        }

        Tensor d = new Tensor(IFM2D, 2, 2, 6);
        Tensor output = l.forward(d);

        assertEquals(output.size(), OFM3IFM2D.length);
        for (int i = 0; i < OFM3IFM2D.length; ++i)
        {
            assertEquals(output.get(i), OFM3IFM2D[i], 1e-6);
        }
        //printRowMajor(((DenseVectorN) output).getX(), 3, 4, 2);
    }

    @Test
    public void testBackward2to3() throws Exception
    {
        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(3, 2, 3, 2);

        Tensor weights = l.getParams();
        for (int i = 0; i < IFM2OFM3K.length; ++i)
        {
            weights.set(i, IFM2OFM3K[i]);
        }

        Tensor d = new Tensor(IFM2D, 2, 2, 6);
        Tensor output = l.forward(d);

        output.scale(1 / 1000.);
        Tensor grads = l.backward(output, 0);

        Tensor gw = l.getParamGrads();
        // Are gradients right?
        double acc = 0.;
        // Are weights right after gradients applied?
        double accW = 0.;
        for (int i = 0, gsz = gw.size(); i < gsz; ++i)
        {
            acc += gw.get(i) * gw.get(i);
            weights.addi(i, gw.get(i));
            accW += weights.get(i) * weights.get(i);//[i];
            gw.set(i, 0);
        }
        assertEquals(SQ_M_1000, acc, 1e-6);
        assertEquals(SQ_M_W_1000, accW, 1e-6);
        assertEquals(grads.size(), OFM3IFM2G_1000.length);

        for (int i = 0; i < OFM3IFM2G_1000.length; ++i)
        {
            assertEquals(OFM3IFM2G_1000[i], grads.at(i), 1e-6);
        }
    }
    @Test
    public void testBackward2to3WordVecAsInChannels() throws Exception
    {
        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(3, 2, 3, 1);

        Tensor weights = l.getParams();
        for (int i = 0; i < IFM2OFM3KNOE.length; ++i)
        {
            weights.set(i, IFM2OFM3KNOE[i]);
        }

        Tensor d = new Tensor(IFM2DNOE, 2, 1, 6);
        Tensor output = l.forward(d);

        output.scale(1 / 1000.);

        Tensor grads = l.backward(output, 0);

        Tensor gw = l.getParamGrads();
        // Are gradients right?
        double acc = 0.;
        // Are weights right after gradients applied?
        double accW = 0.;
        for (int i = 0, gsz = gw.size(); i < gsz; ++i)
        {
            acc += gw.get(i)*gw.get(i);
            //weights.addi(i, gw.get(i));
            double wU = weights.get(i) + gw.get(i);
            accW += wU*wU;
            gw.set(i, 0);
        }
        assertEquals(SQ_M_1000_1CHAN, acc, 1e-6);
        assertEquals(SQ_M_W_1000_1CHAN, accW, 1e-6);
        assertEquals(grads.size(), OFM3IFM2G_1000_1CHAN.length);

        for (int i = 0; i < OFM3IFM2G_1000_1CHAN.length; ++i)
        {
            assertEquals(OFM3IFM2G_1000_1CHAN[i], grads.get(i), 1e-3);
        }
    }
}