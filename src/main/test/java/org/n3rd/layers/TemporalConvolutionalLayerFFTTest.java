package org.n3rd.layers;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;

public class TemporalConvolutionalLayerFFTTest
{

    // embeddings are
    double[] D =
            {
                    1,3,5,7,9,11,
                    2,4,6,8,10,12
                    //1,2,
                    //3,4,
                    //5,6,
                    //7,8,
                    //9,10,
                    //11,12
            };


    double[] K = {
            1,2,3,
            4,5,6
            //1,4,
            //2,5,
            //3,6
            };


    double[] OFM1IFM1 = {22,34,46,58,64,94,124,154};

    double[] IFM2K = {
            1,2,3,
            7,8,9,
            4,5,6,
            10,11,12
            //1,7,
            //2,8,
            //3,9,
            //4,10,
            //5,11,
            //6,12
            };

    double[] OFM1IFM2 = {
            78.5, 120.5, 162.5, 204.5,
            252.5, 366.5, 480.5, 594.5
    };
            //78.5, 252.5,
            //120.5, 366.5,
            //162.5, 480.5,
            //204.5, 594.5};

    double[] IFM2D = {
            1,3,5,7,9,11,
            2,4,6,8,10,12,
            1.5,3.5,5.5,7.5,9.5,11.5,
            2.5,4.5,6.5,8.5,10.5,12.5};

    double[] IFM2DODD = {
            1,3,5,7,9,11,13,
            2,4,6,8,10,12,14,
            1.5,3.5,5.5,7.5,9.5,11.5,13.5,
            2.5,4.5,6.5,8.5,10.5,12.5,14.5};

    double[] IFM2OFM3K = {

            1,2,3,
            7,8,9,

            4,5,6,
            10,11,12,



            1.5,2.5,3.5,
            7.5,8.5,9.5,

            4.5,5.5,6.5,
            10.5,11.5,12.5,



            1.2,2.2,3.2,
            7.2,8.2,9.2,

            4.2,5.2,6.2,
            10.2,11.2,12.2
    };
    /*
            1,2,
            3,4,
            5,6,
            7,8,
            9,10,
            11,12,
            1.5,2.5,
            3.5,4.5,
            5.5,6.5,
            7.5,8.5,
            9.5,10.5,
            11.5,12.5};*/


/*
    double[] IFM2OFM3K = {1,7,   1.5,7.5,   1.2,7.2,
                          2,8,   2.5,8.5,   2.2,8.2,
                          3,9,   3.5,9.5,   3.2,9.2,
                          4,10,  4.5,10.5,  4.2,10.2,
                          5,11,  5.5,11.5,  5.2,11.2,
                          6,12,  6.5,12.5,  6.2,12.2 };

    double[] IFM2OFM3K = {


        1,7,
        2,8,
        3,9,

        4,10,
        5,11,
        6,12,



        1.5,7.5,
        2.5,8.5,
        3.5,9.5,

        4.5,10.5,
        5.5,11.5,
        6.5,12.5,



        1.2,7.2,
        2.2,8.2,
        3.2,9.2,

        4.2,10.2,
        5.2,11.2,
        6.2,12.2 };
*/



            /*// 3,2,3,2
    double[] IFM2OFM3K = {1,7,2,8,3,9,4,10,5,11,6,12,
                          1.5,7.5,2.5,8.5,3.5,9.5,4.5,10.5,5.5,11.5,6.5,12.5,
                          1.2,7.2,2.2,8.2,3.2,9.2,4.2,10.2,5.2,11.2,6.2,12.2 };

double[] OFM3IFM2D = {
                    78.5, 252.5,
                    120.5, 366.5,
                    162.5, 480.5,
                    204.5, 594.5,

                    88.25, 265.25,
                    136.25, 385.25,
                    184.25, 505.25,
                    232.25, 625.25,

                    82.4, 257.6,
                    126.8, 374.0,
                    171.2, 490.4,
                    215.6, 606.8};
*/
    double[] OFM3IFM2D = {
                    78.5,120.5,162.5,204.5,
                    252.5,366.5,480.5,594.5,

                    88.25,136.25,184.25,232.25,
                    265.25,385.25,505.25,625.25,

                    82.4,126.8,171.2,215.6,
                    257.6,374.0,490.4,606.8
                   };

    double[] OFM3IFM2G_1000 = {

            0.30975499999,1.03594,2.312955,3.217995,3.14416,2.116295,
            5.611595,14.53462,27.119474999,35.778915,28.682439999,16.872934999,

            1.057205,2.93404,5.764905,7.879545,6.65506,4.073345,
            7.937645,20.23792, 37.251224999,49.064265,38.59054,22.352585
    };
/*
        double[] OFM3IFM2G_1000 = {
            0.30975499999, 5.611595,
            1.03594, 14.53462,
            2.312955, 27.119474999,
            3.217995, 35.778915,
            3.1441600000, 28.682439999,
            2.116295, 16.872934999,

            1.057205, 7.937645,
            2.93404, 20.23792,
            5.7649050000, 37.251224999,
            7.8795450000, 49.064265,
            6.6550600000, 38.59054,
            4.0733450000, 22.352585
    };*/
    double SQ_M_1000 = 3886.2073516200003;
    double SQ_M_W_1000 = 11477.130271620003;


    @Test
    public void testForward() throws Exception
    {
        TemporalConvolutionalLayerFFT l = new TemporalConvolutionalLayerFFT(1, 1, 3, 2);
        for (int i = 0; i < K.length; ++i)
        {
            l.weights.set(i, K[i]);
        }
        Tensor d = new Tensor(D, 1, 2, 6);
        Tensor output = l.forward(d);

        assertEquals(output.size(), OFM1IFM1.length);

        for (int i = 0; i < OFM1IFM1.length; ++i)
        {
            assertEquals(output.get(i), OFM1IFM1[i], 1e-6);
        }

    }

    @Test
    public void testForward2to1() throws Exception
    {

        TemporalConvolutionalLayerFFT l = new TemporalConvolutionalLayerFFT(1, 2, 3, 2);

        Tensor weights = l.getParams();
        for (int i = 0; i < IFM2K.length; ++i)
        {
            weights.set(i, IFM2K[i]);
        }

        Tensor d = new Tensor(IFM2D, 2, 2, 6);
        Tensor output = l.forward(d);

        for (int i = 0; i < OFM1IFM2.length; ++i)
        {
            assertEquals(output.get(i), OFM1IFM2[i], 1e-6);
        }

    }

    @Test
    public void testForward2to3() throws Exception
    {

        TemporalConvolutionalLayerFFT l = new TemporalConvolutionalLayerFFT(3, 2, 3, 2);

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
    public void testMultiIterSxS() throws Exception
    {
        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(3, 2, 3, 2);

        TemporalConvolutionalLayerFFT lfft = new TemporalConvolutionalLayerFFT(3, 2, 3, 2);

        Tensor weights = l.getParams();
        Tensor weightsFFT = lfft.getParams();
        for (int i = 0; i < IFM2OFM3K.length; ++i)
        {
            weights.set(i, IFM2OFM3K[i]);
            weightsFFT.set(i, IFM2OFM3K[i]);
        }

        Tensor d = new Tensor(IFM2D, 2, 2, 6);

        for (int i = 0; i < 10; ++i)
        {
            for (int j = 0; j < d.size(); ++j)
            {
                d.set(j, d.get(j) + Math.random() - 0.5);
            }
            Tensor output = l.forward(d);
            Tensor outputFFT = lfft.forward(d);
            output.scale(1 / 1000.);
            outputFFT.scale(1 / 1000.);
            Tensor grads = l.backward(output, 0);
            Tensor gradsFFT = lfft.backward(outputFFT, 0);
            Tensor gw = l.getParamGrads();
            Tensor gwfft = lfft.getParamGrads();

            assertEquals(grads.size(), gradsFFT.size());
            for (int k = 0; k < grads.size(); ++k)
            {
                assertEquals(grads.get(k), gradsFFT.get(k), 1e-6);
            }
            for (int k = 0; k < gw.size(); ++k)
            {

                weights.addi(k, gw.get(k));
                gw.set(k, 0);
                weightsFFT.addi(k, gwfft.get(k));
                gwfft.set(k, 0);
            }

        }
        for (int i = 0; i < weights.size(); ++i)
        {
            assertEquals(weights.get(i), weightsFFT.get(i), 1e-6);
        }
    }


    @Test
    public void testSxS() throws Exception
    {
        TemporalConvolutionalLayer l = new TemporalConvolutionalLayer(3, 2, 3, 2);

        TemporalConvolutionalLayerFFT lfft = new TemporalConvolutionalLayerFFT(3, 2, 3, 2);

        Tensor weights = l.getParams();
        Tensor weightsFFT = lfft.getParams();
        for (int i = 0; i < IFM2OFM3K.length; ++i)
        {
            weights.set(i, IFM2OFM3K[i]);
            weightsFFT.set(i, IFM2OFM3K[i]);
        }

        Tensor d = new Tensor(IFM2DODD, 2, 2, 7);

        Tensor output = l.forward(d);
        Tensor outputFFT = lfft.forward(d);
        output.scale(1 / 1000.);
        outputFFT.scale(1 / 1000.);
        Tensor grads = l.backward(output, 0);
        Tensor gradsFFT = lfft.backward(outputFFT, 0);
        Tensor gw = l.getParamGrads();
        Tensor gwfft = lfft.getParamGrads();

        assertEquals(grads.size(), gradsFFT.size());
        for (int k = 0; k < grads.size(); ++k)
        {
            assertEquals(grads.get(k), gradsFFT.get(k), 1e-6);
        }
        for (int k = 0; k < gw.size(); ++k)
        {

            weights.addi(k, gw.get(k));
            gw.set(k, 0);
            weightsFFT.addi(k, gwfft.get(k));
            gwfft.set(k, 0);
        }
        for (int i = 0; i < weights.size(); ++i)
        {
            assertEquals(weights.get(i), weightsFFT.get(i), 1e-6);
        }
    }


    @Test
    public void testBackward2to3() throws Exception
    {
        TemporalConvolutionalLayerFFT l = new TemporalConvolutionalLayerFFT(3, 2, 3, 2);

        Tensor weights = l.getParams();
        for (int i = 0; i < IFM2OFM3K.length; ++i)
        {
            weights.set(i, IFM2OFM3K[i]);
        }

        Tensor d = new Tensor(IFM2D, 2, 2, 6);
        Tensor output = l.forward(d);

        int sz = output.size();
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
        assertEquals(SQ_M_1000, acc, 1e-6);
        assertEquals(SQ_M_W_1000, accW, 1e-6);
        assertEquals(grads.size(), OFM3IFM2G_1000.length);

        for (int i = 0; i < OFM3IFM2G_1000.length; ++i)
        {
            assertEquals(OFM3IFM2G_1000[i], grads.get(i), 1e-6);
        }
        output = l.forward(d);
        output.scale(1 / 1000.);
        grads = l.backward(output, 0);
        accW = 0.;
        acc = 0.;
        for (int i = 0, gsz = gw.size(); i < gsz; ++i)
        {
            acc += gw.get(i)*gw.get(i);
            double wU = weights.get(i) + gw.get(i);
            accW += wU*wU;
            gw.set(i, 0);
        }
        assertEquals(SQ_M_1000, acc, 1e-6);
        assertEquals(SQ_M_W_1000, accW, 1e-6);
        assertEquals(grads.size(), OFM3IFM2G_1000.length);

        for (int i = 0; i < OFM3IFM2G_1000.length; ++i)
        {
            assertEquals(OFM3IFM2G_1000[i], grads.get(i), 1e-6);
        }
    }
}