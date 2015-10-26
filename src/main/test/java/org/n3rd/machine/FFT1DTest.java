package org.n3rd.machine;

import com.sun.media.sound.FFT;
import org.junit.Test;
import org.n3rd.ops.FFTOps;
import org.sgdtk.ArrayDouble;

import static junit.framework.TestCase.assertEquals;


public class FFT1DTest
{

    double[] D =
            {
                    1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10,
                    11, 12
            };

    double[] D2 = { 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12 };

    double[] DOFF = { 115, 126, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, -42, -86 };

    double[] K = { 1, 4, 2, 5, 3, 6 };
    double[] KOFF = { -64, -102, 1, 4, 2, 5, 3, 6, -100, -10, -1, 0.9 };

    double[] K2 = { 5, 3, 6 };


    double[] O1 = { 22, 34, 46, 58 };
    double[] O1NEG = {-35,  -58,  -79, -100, -115};

    double[] xcorr(double[] x, double[] y)
    {
        int narrow = x.length - y.length + 1;
        double[] z = new double[narrow];
        for (int i = 0; i < narrow; ++i)
        {

            for (int j = 0; j < y.length; ++j)
            {
                z[i] += x[i + j] * y[j];
            }
        }
        return z;

    }

    double[] fftfilt(double[] x, double[] y, boolean corr)
    {

        final int wide = ArrayDouble.nextPowerOf2(x.length + y.length - 1);
        final int narrow = x.length - y.length + 1;
        double[] z = new double[narrow];

        final FFT fft = new FFT(wide, -1);
        final FFT ifft = new FFT(wide, 1);

        double[] xwide = new double[wide * 2];
        double[] ywide = new double[wide * 2];

        for (int i = 0, j = 0; i < x.length; ++i, j += 2)
        {
            xwide[j] = x[i];
        }

        if (corr)
        {
            for (int i = 0, j = 0; i < y.length; ++i, j += 2)
            {
                ywide[j] = y[i];
            }
        } else
        {
            for (int i = 0, j = 0; i < y.length; ++i, j += 2)
            {
                ywide[j] = y[y.length - i - 1];
            }
        }

        fft.transform(xwide);
        fft.transform(ywide);

        for (int i = 0; i < xwide.length; i += 2)
        {
            ywide[i + 1] = -ywide[i + 1];
            double xwr = xwide[i];
            double xwi = xwide[i + 1];
            xwide[i] = xwr * ywide[i] - xwi * ywide[i + 1];
            xwide[i + 1] = xwr * ywide[i + 1] + xwi * ywide[i];
        }

        ifft.transform(xwide);

        for (int i = 0, j = 0; i < narrow; ++i, j += 2)
        {
            double re = xwide[j] / wide;
            z[i] = re;
        }

        return z;
    }

    @Test
    public void testFFT1() throws Exception
    {

        //Tensor d = new Tensor(D, 1, 6, 2);
        double[] x = { 1, 3, 5, 7, 9, 11 };
        double[] y = { 1, 2, 3 };
        //Tensor k = new Tensor(K, 3, 2);

        double[] z = fftfilt(x, y, true);

        for (int i = 0; i < z.length; ++i)
        {

            assertEquals(z[i], O1[i]);
        }

    }

    @Test
    public void testLongSignal() throws Exception
    {
        double[] x = new double[2140];
        for (int i = 0; i < x.length; ++i)
        {
            x[i] = i;
        }
        double[] y = new double[258];
        for (int i = 0; i < y.length; ++i)
        {
            y[i] = i;
        }

        double[] z1 = null;
        double[] z2 = null;
        double t0 = System.currentTimeMillis();
        for (int i = 0; i < 2000; ++i)
        {
            z1 = xcorr(x, y);
        }
        double xe = System.currentTimeMillis() - t0;
        System.out.println("Xcorr speed: " + xe  + "ms");
        t0 = System.currentTimeMillis();
        for (int i = 0; i < 2000; ++i)
        {
            z2 = fftfilt(x, y, true);
        }
        double xf = System.currentTimeMillis() - t0;

        System.out.println("FFT speed:   " + xf  + "ms");

        assertEquals(z1.length, z2.length);
        for (int i = 0; i < z1.length; ++i)
        {
            assertEquals(z1[i], z2[i], 1e-4);
        }


    }
    @Test
    public void testFFT1Neg() throws Exception
    {

        //Tensor d = new Tensor(D, 1, 6, 2);
        double[] x = { -1, 2, -3, 4, 5, 6, 7, 8, 9 };
        double[] y = { -1, -2, -3, -4, -5 };
        //Tensor k = new Tensor(K, 3, 2);

        double[] z = fftfilt(x, y, true);

        for (int i = 0; i < z.length; ++i)
        {

            assertEquals(z[i], O1NEG[i], 1e-6);
        }

    }

    @Test
    public void testFFTOps() throws Exception
    {

        //Tensor d = new Tensor(D, 1, 6, 2);
        double[] x = { 1, 3, 5, 7, 9, 11 };
        double[] y = { 1, 2, 3 };
        //Tensor k = new Tensor(K, 3, 2);

        double[] z = new double[x.length - y.length + 1];
        new FFTOps().filter(x, 0, x.length, y, 0, y.length, z, true);

        for (int i = 0; i < z.length; ++i)
        {
            assertEquals(z[i], O1[i]);
        }
    }

    @Test
    public void testFFTOpsPad() throws Exception
    {

        //Tensor d = new Tensor(D, 1, 6, 2);

        //Tensor k = new Tensor(K, 3, 2);

        double[] zcorr = new double[D2.length - K.length + 1];
        new FFTOps().filter(D2, 0, D2.length, K, 0, K.length, zcorr, true);

        double[] z = new double[D.length - K.length + 1];
        new FFTOps().filter(DOFF, 2, D2.length, KOFF, 2, K.length, z, true);

        for (int i = 0; i < z.length; ++i)
        {
            assertEquals(z[i], zcorr[i]);
        }
    }


    @Test
    public void testFFTComp() throws Exception
    {
        double[] x = { 1, 3, 5, 7, 9, 11 };
        double[] y = { 1, 2, 3 };
        double[] z = new double[D2.length - K2.length + 1];
        FFTOps fft = new FFTOps();
        //double [] z = new double[x.length - y.length + 1];
        fft.filter(x, 0, x.length, y, 0, y.length, z, true);

        for (int i = 0; i < x.length - y.length + 1; ++i)
        {
            assertEquals(z[i], O1[i]);

        }
        fft.filter(x, 0, x.length, y, 0, y.length, z, true);


        for (int i = 0; i < x.length - y.length + 1; ++i)
        {
            assertEquals(z[i], O1[i]);

        }

        fft.filter(D2, 0, D2.length, K2, 0, K2.length, z, true);

        double[] zcorr = fftfilt(D2, K2, true);
        assertEquals(z.length, zcorr.length);
        for (int i = 0; i < z.length; ++i)
        {
            assertEquals(z[i], zcorr[i]);
        }

        z = new double[x.length - y.length + 1];

        fft.filter(x, 0, x.length, y, 0, y.length, z, true);


        for (int i = 0; i < x.length - y.length + 1; ++i)
        {
            assertEquals(z[i], O1[i]);

        }
    }

    @Test
    public void testMany() throws Exception
    {

        double[] z = new double[D2.length - K2.length + 1];

        FFTOps fft = new FFTOps();

        for (int it = 0; it < 1000; ++it)
        {
            int rEnd = (int) (Math.random() * D2.length);
            int rStart = (int) (Math.random() * rEnd);
            while (rEnd - rStart < K2.length)
            {
                rStart--;
            }

            if (rStart < 0)
            {
                rStart = 0;
                rEnd = D2.length;

            }

            double[] x = new double[rEnd - rStart];
            for (int i = rStart, j = 0; i < rEnd; ++i, ++j)
            {
                x[j] = D2[i];
            }

            fft.filter(x, 0, x.length, K2, 0, K2.length, z, true);

            double[] zcorr = fftfilt(x, K2, true);
            for (int i = 0; i < zcorr.length; ++i)
            {
                assertEquals(z[i], zcorr[i]);

            }
        }

    }

    @Test
    public void testFFTOpsSpeed() throws Exception
    {

        double[] z = new double[D2.length - K2.length + 1];

        FFTOps fft = new FFTOps();


        double t0 = System.currentTimeMillis();

        for (int it = 0; it < 100000; ++it)
        {
            int rEnd = (int) (Math.random() * D2.length);
            int rStart = (int) (Math.random() * rEnd);
            while (rEnd - rStart < K2.length)
            {
                rStart--;
            }

            if (rStart < 0)
            {
                rStart = 0;
                rEnd = D2.length;

            }

            double[] x = new double[rEnd - rStart];
            for (int i = rStart, j = 0; i < rEnd; ++i, ++j)
            {
                x[j] = D2[i];
            }

            fft.filter(x, 0, x.length, K2, 0, K2.length, z, true);

        }
        double t = System.currentTimeMillis() - t0;
        System.out.println("Class exec time: " + t + "ms");

    }

    @Test
    public void testFFTFnSpeed() throws Exception
    {

        double t0 = System.currentTimeMillis();

        for (int it = 0; it < 100000; ++it)
        {
            int rEnd = (int) (Math.random() * D2.length);
            int rStart = (int) (Math.random() * rEnd);
            while (rEnd - rStart < K2.length)
            {
                rStart--;
            }

            if (rStart < 0)
            {
                rStart = 0;
                rEnd = D2.length;

            }

            double[] x = new double[rEnd - rStart];
            for (int i = rStart, j = 0; i < rEnd; ++i, ++j)
            {
                x[j] = D2[i];
            }

            double[] z = fftfilt(x, K2, true);

        }
        double t = System.currentTimeMillis() - t0;
        System.out.println("Class exec time: " + t + "ms");

    }

}
