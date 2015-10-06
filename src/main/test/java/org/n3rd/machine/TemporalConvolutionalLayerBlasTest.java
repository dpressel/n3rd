package org.n3rd.machine;

import org.junit.Test;
import org.n3rd.Tensor;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertNotSame;

public class TemporalConvolutionalLayerBlasTest
{

    double[] IFM2D = { 1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            1.5, 2.5,
            3.5, 4.5,
            5.5, 6.5,
            7.5, 8.5,
            9.5, 10.5,
            11.5, 12.5 };
    // Kernel for embedding one should be 1,2,3 and kernel for embedding two should be 4,5,6
    // for non-blas version, the memory is laid out as (nK, kL, kW, eSz)


    double[] D =
            {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10,
                    11, 12
            };

    /* Equivalent non-unrolled
    double[] K = {1,4,
                  2,5,
                  3,6};
    */
    double[] IFM2K = { 1, 7,
            2, 8,
            3, 9,
            4, 10,
            5, 11,
            6, 12 };

    double[] OFM1IFM2 = { 78.5, 252.5,
            120.5, 366.5,
            162.5, 480.5,
            204.5, 594.5 };

    double[] OFM1IFM1 = { 22, 64, 34, 94, 46, 124, 58, 154 };


    double[] IFM2OFM3K = { 1, 7, 1.5, 7.5, 1.2, 7.2,
            2, 8, 2.5, 8.5, 2.2, 8.2,
            3, 9, 3.5, 9.5, 3.2, 9.2,
            4, 10, 4.5, 10.5, 4.2, 10.2,
            5, 11, 5.5, 11.5, 5.2, 11.2,
            6, 12, 6.5, 12.5, 6.2, 12.2 };

    double[] OFM3IFM2D = { 78.5,
            252.5,
            120.5,
            366.5,
            162.5,
            480.5,
            204.5,
            594.5,
            88.25,
            265.25,
            136.25,
            385.25,
            184.25,
            505.25,
            232.25,
            625.25,
            82.4,
            257.6,
            126.8,
            374.0,
            171.2,
            490.4,
            215.6,
            606.8 };

    double[] OFM3IFM2G_1000 = {
            0.30975499999,
            5.611595,
            1.03594,
            14.53462,
            2.312955,
            27.119474999,
            3.217995,
            35.778915,
            3.1441600000,
            28.682439999,
            2.116295,
            16.872934999,
            1.057205,
            7.937645,
            2.93404,
            20.23792,
            5.7649050000,
            37.251224999,
            7.8795450000,
            49.064265,
            6.6550600000,
            38.59054,
            4.0733450000,
            22.352585
    };
    double SQ_M_1000 = 3886.2073516200003;
    double SQ_M_W_1000 = 11477.130271620003;
    @Test
    public void testForward() throws Exception
    {
        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(1, 1, 3, 2);
        Tensor weights = l.getParams();
        assertNotSame(weights.d[0], 0.0);
        assertNotSame(weights.d[1], 0.0);
        assertNotSame(weights.d[2], 0.0);
        weights.d[0] = 1;
        weights.d[1] = 2;
        weights.d[2] = 3;
        assertEquals(weights.d[3], 0.0);
        assertEquals(weights.d[4], 0.0);
        assertEquals(weights.d[5], 0.0);
        assertEquals(weights.d[6], 0.0);
        assertEquals(weights.d[7], 0.0);
        assertEquals(weights.d[8], 0.0);
        assertNotSame(weights.d[9], 0.0);
        assertNotSame(weights.d[10], 0.0);
        assertNotSame(weights.d[11], 0.0);
        weights.d[9] = 4;
        weights.d[10] = 5;
        weights.d[11] = 6;

        Tensor d = new Tensor(D, 1, 6, 2);
        Tensor output = l.forward(d);

        double[] x = output.d;
        for (int i = 0; i < OFM1IFM1.length; ++i)
        {
            System.out.println(x[i]);
            assertEquals(x[i], OFM1IFM1[i]);
        }

    }

    @Test
    public void testForward2to1() throws Exception
    {


        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(1, 2, 3, 2);
        Tensor weights = l.getParams();
        assertNotSame(weights.d[0], 0.0);
        assertNotSame(weights.d[1], 0.0);
        assertNotSame(weights.d[2], 0.0);
        assertNotSame(weights.d[3], 0.0);
        assertNotSame(weights.d[4], 0.0);
        assertNotSame(weights.d[5], 0.0);
        weights.d[0] = 1;
        weights.d[1] = 2;
        weights.d[2] = 3;
        weights.d[3] = 4;
        weights.d[4] = 5;
        weights.d[5] = 6;
        assertEquals(weights.d[6], 0.0);
        assertEquals(weights.d[7], 0.0);
        assertEquals(weights.d[8], 0.0);
        assertEquals(weights.d[9], 0.0);
        assertEquals(weights.d[10], 0.0);
        assertEquals(weights.d[11], 0.0);

        assertEquals(weights.d[12], 0.0);
        assertEquals(weights.d[13], 0.0);
        assertEquals(weights.d[14], 0.0);
        assertEquals(weights.d[15], 0.0);
        assertEquals(weights.d[16], 0.0);
        assertEquals(weights.d[17], 0.0);


        assertNotSame(weights.d[18], 0.0);
        assertNotSame(weights.d[19], 0.0);
        assertNotSame(weights.d[20], 0.0);
        assertNotSame(weights.d[21], 0.0);
        assertNotSame(weights.d[22], 0.0);
        assertNotSame(weights.d[23], 0.0);
        weights.d[18] = 7;
        weights.d[19] = 8;
        weights.d[20] = 9;
        weights.d[21] = 10;
        weights.d[22] = 11;
        weights.d[23] = 12;

        Tensor d = new Tensor(IFM2D, 2, 6, 2);
        Tensor output = l.forward(d);


        double[] x = output.d;

        for (int i = 0; i < OFM1IFM2.length; ++i)
        {
            System.out.println(x[i]);
            assertEquals(x[i], OFM1IFM2[i]);
        }

    }


    @Test
    public void testForward2to3() throws Exception
    {


        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(3, 2, 3, 2);
        Tensor weights = l.getParams();
        // K1

        assertNotSame(weights.d[0], 0.0);
        assertNotSame(weights.d[1], 0.0);
        assertNotSame(weights.d[2], 0.0);
        assertNotSame(weights.d[3], 0.0);
        assertNotSame(weights.d[4], 0.0);
        assertNotSame(weights.d[5], 0.0);
        weights.d[0] = 1;
        weights.d[1] = 2;
        weights.d[2] = 3;
        weights.d[3] = 4;
        weights.d[4] = 5;
        weights.d[5] = 6;
        assertEquals(weights.d[6], 0.0);
        assertEquals(weights.d[7], 0.0);
        assertEquals(weights.d[8], 0.0);
        assertEquals(weights.d[9], 0.0);
        assertEquals(weights.d[10], 0.0);
        assertEquals(weights.d[11], 0.0);

        assertEquals(weights.d[12], 0.0);
        assertEquals(weights.d[13], 0.0);
        assertEquals(weights.d[14], 0.0);
        assertEquals(weights.d[15], 0.0);
        assertEquals(weights.d[16], 0.0);
        assertEquals(weights.d[17], 0.0);


        assertNotSame(weights.d[18], 0.0);
        assertNotSame(weights.d[19], 0.0);
        assertNotSame(weights.d[20], 0.0);
        assertNotSame(weights.d[21], 0.0);
        assertNotSame(weights.d[22], 0.0);
        assertNotSame(weights.d[23], 0.0);
        weights.d[18] = 7;
        weights.d[19] = 8;
        weights.d[20] = 9;
        weights.d[21] = 10;
        weights.d[22] = 11;
        weights.d[23] = 12;

        // K2

        assertNotSame(weights.d[24], 0.0);
        assertNotSame(weights.d[25], 0.0);
        assertNotSame(weights.d[26], 0.0);
        assertNotSame(weights.d[27], 0.0);
        assertNotSame(weights.d[28], 0.0);
        assertNotSame(weights.d[29], 0.0);
        weights.d[24] = 1 + 0.5;
        weights.d[25] = 2 + 0.5;
        weights.d[26] = 3 + 0.5;
        weights.d[27] = 4 + 0.5;
        weights.d[28] = 5 + 0.5;
        weights.d[29] = 6 + 0.5;
        assertEquals(weights.d[30], 0.0);
        assertEquals(weights.d[31], 0.0);
        assertEquals(weights.d[32], 0.0);
        assertEquals(weights.d[33], 0.0);
        assertEquals(weights.d[34], 0.0);
        assertEquals(weights.d[35], 0.0);

        assertEquals(weights.d[36], 0.0);
        assertEquals(weights.d[37], 0.0);
        assertEquals(weights.d[38], 0.0);
        assertEquals(weights.d[39], 0.0);
        assertEquals(weights.d[40], 0.0);
        assertEquals(weights.d[41], 0.0);


        assertNotSame(weights.d[42], 0.0);
        assertNotSame(weights.d[43], 0.0);
        assertNotSame(weights.d[44], 0.0);
        assertNotSame(weights.d[45], 0.0);
        assertNotSame(weights.d[46], 0.0);
        assertNotSame(weights.d[47], 0.0);
        weights.d[42] = 7 + 0.5;
        weights.d[43] = 8 + 0.5;
        weights.d[44] = 9 + 0.5;
        weights.d[45] = 10 + 0.5;
        weights.d[46] = 11 + 0.5;
        weights.d[47] = 12 + 0.5;

        // K3
        assertNotSame(weights.d[48], 0.0);
        assertNotSame(weights.d[49], 0.0);
        assertNotSame(weights.d[50], 0.0);
        assertNotSame(weights.d[51], 0.0);
        assertNotSame(weights.d[52], 0.0);
        assertNotSame(weights.d[53], 0.0);
        weights.d[48] = 1 + 0.2;
        weights.d[49] = 2 + 0.2;
        weights.d[50] = 3 + 0.2;
        weights.d[51] = 4 + 0.2;
        weights.d[52] = 5 + 0.2;
        weights.d[53] = 6 + 0.2;
        assertEquals(weights.d[54], 0.0);
        assertEquals(weights.d[55], 0.0);
        assertEquals(weights.d[56], 0.0);
        assertEquals(weights.d[57], 0.0);
        assertEquals(weights.d[58], 0.0);
        assertEquals(weights.d[59], 0.0);

        assertEquals(weights.d[60], 0.0);
        assertEquals(weights.d[61], 0.0);
        assertEquals(weights.d[62], 0.0);
        assertEquals(weights.d[63], 0.0);
        assertEquals(weights.d[64], 0.0);
        assertEquals(weights.d[65], 0.0);


        assertNotSame(weights.d[66], 0.0);
        assertNotSame(weights.d[67], 0.0);
        assertNotSame(weights.d[68], 0.0);
        assertNotSame(weights.d[69], 0.0);
        assertNotSame(weights.d[70], 0.0);
        assertNotSame(weights.d[71], 0.0);
        weights.d[66] = 7 + 0.2;
        weights.d[67] = 8 + 0.2;
        weights.d[68] = 9 + 0.2;
        weights.d[69] = 10 + 0.2;
        weights.d[70] = 11 + 0.2;
        weights.d[71] = 12 + 0.2;

        Tensor d = new Tensor(IFM2D, 2, 6, 2);
        Tensor output = l.forward(d);

        assertEquals(output.size(), OFM3IFM2D.length);
        for (int i = 0; i < OFM3IFM2D.length; ++i)
        {
            assertEquals(output.d[i], OFM3IFM2D[i], 1e-6);

        }

    }

    @Test
    public void testBackward2to3() throws Exception
    {

        TemporalConvolutionalLayerBlas l = new TemporalConvolutionalLayerBlas(3, 2, 3, 2);
        Tensor weights = l.getParams();

        // K1
        weights.d[0] = 1;
        weights.d[1] = 2;
        weights.d[2] = 3;
        weights.d[3] = 4;
        weights.d[4] = 5;
        weights.d[5] = 6;
        weights.d[18] = 7;
        weights.d[19] = 8;
        weights.d[20] = 9;
        weights.d[21] = 10;
        weights.d[22] = 11;
        weights.d[23] = 12;
        //K2
        weights.d[24] = 1 + 0.5;
        weights.d[25] = 2 + 0.5;
        weights.d[26] = 3 + 0.5;
        weights.d[27] = 4 + 0.5;
        weights.d[28] = 5 + 0.5;
        weights.d[29] = 6 + 0.5;
        weights.d[42] = 7 + 0.5;
        weights.d[43] = 8 + 0.5;
        weights.d[44] = 9 + 0.5;
        weights.d[45] = 10 + 0.5;
        weights.d[46] = 11 + 0.5;
        weights.d[47] = 12 + 0.5;
        weights.d[48] = 1 + 0.2;
        weights.d[49] = 2 + 0.2;
        weights.d[50] = 3 + 0.2;
        weights.d[51] = 4 + 0.2;
        weights.d[52] = 5 + 0.2;
        weights.d[53] = 6 + 0.2;
        weights.d[66] = 7 + 0.2;
        weights.d[67] = 8 + 0.2;
        weights.d[68] = 9 + 0.2;
        weights.d[69] = 10 + 0.2;
        weights.d[70] = 11 + 0.2;
        weights.d[71] = 12 + 0.2;

        Tensor d = new Tensor(IFM2D, 2, 6, 2);
        Tensor ograd = l.forward(d);

        for (int i = 0, sz = ograd.size(); i < sz; ++i)
        {

            ograd.d[i] /= 1000.;
        }
        Tensor grads = l.backward(ograd, 0);


        Tensor gw = l.getParamGrads();
        // Are gradients right?
        double acc = 0.;
        // Are weights right after gradients applied?
        double accW = 0.;
        for (int i = 0, sz = gw.size(); i < sz; ++i)
        {
            acc += gw.d[i]*gw.d[i];
            weights.d[i] += gw.d[i];
            accW += weights.d[i] * weights.d[i];
            gw.d[i] = 0;
        }
        assertEquals(SQ_M_1000, acc, 1e-6);
        System.out.println(accW);
        assertEquals(SQ_M_W_1000, accW, 1e-6);

        for (int i = 0, sz = grads.size(); i < sz; ++i)
        {
            //System.out.println(dx[i]);
            assertEquals(OFM3IFM2G_1000[i], grads.d[i], 1e-6);
        }



    }


}
