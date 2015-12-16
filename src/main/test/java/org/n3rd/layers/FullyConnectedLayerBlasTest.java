package org.n3rd.layers;

import org.junit.Test;
import org.n3rd.Tensor;
import org.n3rd.layers.FullyConnectedLayer;
import org.n3rd.layers.FullyConnectedLayerBlas;

import static junit.framework.TestCase.assertEquals;

public class FullyConnectedLayerBlasTest
{
    final static double[] K = { 1, 2, 3, 4, 5, 6, 7, 8 };
    final static int M = 2;
    final static int N = 4;
    final static double[] V_4 = { 0.4, 0.3, 0.2, 0.1 };
    final static double[] V_2 = { 2, 6 };

    @Test
    public void testForward() throws Exception
    {
        FullyConnectedLayer fc = new FullyConnectedLayer(2, 4);
        Tensor w = fc.getParams();
        double[] b = fc.getBiasParams();
        assertEquals(K.length, w.size());
        FullyConnectedLayerBlas nfc = new FullyConnectedLayerBlas(2, 4);
        Tensor nw = nfc.getParams();
        double[] nb = nfc.getBiasParams();

        assertEquals(K.length, nw.size());

        int n = 0;
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int idx = i * N + j;
                int cIdx = j * M + i;
                nw.set(cIdx, K[n]);
                w.set(idx, K[n]);
                n++;
            }
        }

        for (int i = 0; i < b.length; ++i)
        {
            b[i] = nb[i] = 0.;
        }


        Tensor d = new Tensor(V_4, V_4.length);
        Tensor o = fc.forward(d);
        Tensor no = nfc.forward(d);
        assertEquals(o.size(), V_2.length);
        assertEquals(no.size(), V_2.length);

        for (int i = 0; i < V_2.length; ++i)
        {
            assertEquals(no.get(i), V_2[i]);
            assertEquals(o.get(i), V_2[i]);
        }
    }

    @Test
    public void testBackward4to2() throws Exception
    {
        FullyConnectedLayer fc = new FullyConnectedLayer(2, 4);
        Tensor w = fc.getParams();
        double[] b = fc.getBiasParams();
        assertEquals(K.length, w.size());
        FullyConnectedLayerBlas nfc = new FullyConnectedLayerBlas(2, 4);
        Tensor nw = nfc.getParams();
        double[] nb = nfc.getBiasParams();

        assertEquals(K.length, nw.size());


        int n = 0;
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int idx = i * N + j;
                int cIdx = j * M + i;
                nw.set(cIdx, K[n]);
                w.set(idx, K[n]);
                n++;
            }
        }


        for (int i = 0; i < b.length; ++i)
        {
            b[i] = nb[i] = 0.;
        }

        Tensor error = new Tensor(V_2, V_2.length);
        Tensor d = new Tensor(V_4, V_4.length);
        fc.forward(d);
        Tensor v = fc.backward(error, 0.);

        nfc.forward(d);
        Tensor nv = nfc.backward(error, 0.);

        for (int i = 0; i < v.size(); ++i)
        {
            System.out.print(v.get(i) + " ");
            assertEquals(v.get(i), nv.get(i));
        }
        System.out.println();


        Tensor gw = fc.getParamGrads();
        Tensor ngw = nfc.getParamGrads();

        assertEquals(gw.size(), ngw.size());

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                assertEquals(ngw.get(j * M + i), gw.get(i * N + j), 1e-6);
                System.out.print(gw.get(i * N + j) + " ");
            }
            System.out.println();

        }
        System.out.println();

    }


    @Test
    public void testBackward2to4() throws Exception
    {
        FullyConnectedLayer fc = new FullyConnectedLayer(4, 2);
        Tensor w = fc.getParams();
        double[] b = fc.getBiasParams();
        assertEquals(K.length, w.size());
        FullyConnectedLayerBlas nfc = new FullyConnectedLayerBlas(4, 2);
        Tensor nw = nfc.getParams();
        double[] nb = nfc.getBiasParams();

        assertEquals(K.length, nw.size());

        int n = 0;
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < M; ++j)
            {
                int idx = i * M + j;
                int cIdx = j * N + i;
                nw.set(cIdx, K[n]);
                w.set(idx, K[n]);
                n++;
            }
        }

        for (int i = 0; i < b.length; ++i)
        {
            b[i] = nb[i] = 0.;
        }

        Tensor error = new Tensor(V_4, V_4.length);
        Tensor d = new Tensor(V_2, V_2.length);
        fc.forward(d);
        Tensor v = fc.backward(error, 0.);

        //d = new DenseVectorN(V_2);
        nfc.forward(d);
        Tensor nv = nfc.backward(error, 0.);

        for (int i = 0; i < v.size(); ++i)
        {
            System.out.println(v.get(i) + " ");
            assertEquals(v.get(i), nv.get(i));

        }
        System.out.println();

        Tensor gw = fc.getParamGrads();
        Tensor ngw = nfc.getParamGrads();

        assertEquals(gw.size(), ngw.size());

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < M; ++j)
            {
                assertEquals(ngw.get(j * N + i), gw.get(i * M + j), 1e-6);
                System.out.print(gw.get(i * M + j) + " ");
            }
            System.out.println();

        }
    }

}
