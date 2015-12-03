package org.n3rd.layers;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;

public class TemporalConvolutionalLayerBlasTest
{


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
            -0.14, -0.057, 0.31499999999, 0.873, 1.090999999, 0.8219999, 1.963, 4.953, 9.072, 11.7359999, 9.293, 5.415

    };

    @Test
    public void testForwardWordVecAsInChannels() throws Exception
    {
        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(1, 1, 3);
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

        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(1, 2, 3);

        Tensor weights = l.getParams();

        for (int i = 0; i < IFM2KNOE.length; ++i)
        {
                weights.set(i, IFM2KNOE[i]);

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

        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(3, 2, 3);

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
    public void testBackward2to3WordVecAsInChannels() throws Exception
    {
        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(3, 2, 3);

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
            acc += gw.get(i) * gw.get(i);
            //weights.addi(i, gw.get(i));
            double wU = weights.get(i) + gw.get(i);
            accW += wU * wU;
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