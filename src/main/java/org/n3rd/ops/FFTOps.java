package org.n3rd.ops;

import com.sun.media.sound.FFT;
import org.sgdtk.ArrayDouble;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
/**
 * Convolution and Cross-correlation methods using FFTs.
 * As in the case of Unsafe, the sun media libraries have been ported to
 * open JDKs, so using their APIs should not be an issue here.
 *
 * @author dpressel  
 *
 */
public class FFTOps
{

    double[] xwide;
    double[] ywide;

    Map<Integer, FFT> ffts;
    Map<Integer, FFT> iffts;

    public FFTOps()
    {
        xwide = null;
        ywide = null;
        ffts = new HashMap<Integer, FFT>();
        iffts = new HashMap<Integer, FFT>();

    }

    public void filter(double[] x, int x0, int xLength, double[] y, int y0, int yLength, double[] z, boolean corr)
    {

        final int wide = ArrayDouble.nextPowerOf2(xLength + yLength - 1);
        final int narrow = xLength - yLength + 1;
        assert(z.length >= narrow);

        final int doubleWide = wide*2;
        if (xwide == null || xwide.length <= doubleWide)
        {
            xwide = new double[doubleWide];
            ywide = new double[doubleWide];
        }
        else
        {
            Arrays.fill(xwide, 0);
            Arrays.fill(ywide, 0);
        }

        FFT fft = ffts.get(wide);
        FFT ifft = iffts.get(wide);
        if (fft == null)
        {
            fft = new FFT(wide, -1);
            ffts.put(wide, fft);
        }

        if (ifft == null)
        {
            ifft = new FFT(wide, 1);
            iffts.put(wide, ifft);
        }

        for (int i = 0, j = 0; i < xLength; ++i, j += 2)
        {
            xwide[j] = x[x0 + i];
        }

        if (!corr)
        {
            for (int i = 0, j = 0; i < yLength; ++i, j += 2)
            {
                ywide[j] = y[y0 + i];
            }
        }
        else
        {
            for (int i = 0, j = 0; i < yLength; ++i, j += 2)
            {
                int realEnd = y0 + yLength;
                //int realStart = y0 + i;
                ywide[j] = y[realEnd - i - 1];
            }
        }

        fft.transform(xwide);
        fft.transform(ywide);

        for (int i = 0; i < doubleWide; i+= 2)
        {
            double xwr = xwide[i];
            double xwi = xwide[i+1];
            xwide[i] = xwr*ywide[i] - xwi*ywide[i+1];
            xwide[i+1] = xwr*ywide[i+1] + xwi*ywide[i];
        }

        ifft.transform(xwide);

        for (int i = 0, j = 0; j < doubleWide; ++i, j += 2)
        {
            double re = xwide[j]/wide;
            double im = xwide[j+1]/wide;
            xwide[i] = Math.sqrt(re*re + im*im);
        }

        for (int i = yLength - 1, j = 0; j < narrow; ++i, ++j)
        {
            z[j] = xwide[i];
        }

    }

}
