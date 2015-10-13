package org.n3rd.ops;

import org.jblas.NativeBlas;
import org.n3rd.Tensor;

/**
 * Basically, a bunch of cross-correlation and convolution implementations
 *
 * @author dpressel
 */
public class FilterOps
{
    // Here we are applying the chain gradient from backprop (ygrad) as a cross-corr filter on the
    // input layer.  This (of course) yields a weight gradient surface, which better be the same size as
    // the weights themselves
    public static Tensor corr2Weights(Tensor x, Tensor ygrad)
    {
        // x is then the input, and ygrad is the output, which is usually going to be smaller
        // x.dims[0] is #feature maps in the input
        // x.dims[1] is the #rows in input
        // x.dims[2] is the #cols in input

        final int nFeatureMapsInput = x.dims[0];
        final int xRows = x.dims[1];
        final int xCols = x.dims[2];


        // y is the deltas
        // y.dims[0] is the #feature maps in the output
        final int nFeatureMapsOutput = ygrad.dims[0];
        // y.dims[1] is the #rows in output
        final int yRows = ygrad.dims[1];

        // y.dims[2][ is the #cols in ouptut
        final int yCols = ygrad.dims[2];


        // The number of cubes is the output depth (# feature maps)
        Tensor weightGrads = new Tensor(nFeatureMapsOutput, nFeatureMapsInput, xRows - yRows + 1, xCols - yCols + 1);

        final int kRows = weightGrads.dims[2];
        final int kCols = weightGrads.dims[3];

        // For each feature map
        for (int k = 0; k < nFeatureMapsInput; ++k)
        {
            // The weight gradient is the size of the kernel itself, which is a cube of size input depth x kh x kw
            //int okbase = k * kRows;

            for (int l = 0; l < nFeatureMapsOutput; ++l)
            {

                // For each input row
                for (int i = 0; i < kRows; ++i)
                {
                    //int kbase = (okbase + i) * kCols;
                    // For each input col
                    for (int j = 0; j < kCols; ++j)
                    {

                        int wAddr = ((l * nFeatureMapsInput + k) * kRows + i) * kCols + j;

                        // For input depth
                        double acc = 0;

                        // corr2!!
                        for (int m = 0; m < yRows; ++m)
                        {
                            for (int n = 0; n < yCols; ++n)
                            {
                                int xAddr = (k * xRows + i + m) * xCols + j + n;
                                //final int kh = m;//ygrad.h - m - 1;
                                //final int kw = n;//ygrad.w - n - 1;
                                int yAddr = (l * yRows + m) * yCols + n;
                                acc += x.d[xAddr] * ygrad.d[yAddr];
                            }
                        }

                        weightGrads.d[wAddr] = acc;
                    }
                }
            }
        }
        return weightGrads;
    }

    public static Tensor corr1Weights(Tensor x, Tensor ygrad)
    {
        // x is then the input, and ygrad is the output, which is usually going to be smaller
        // x.dims[0] is #feature maps in the input
        // x.dims[1] is the #rows in input
        // x.dims[2] is the #cols in input

        final int nFeatureMapsInput = x.dims[0];
        final int xRows = x.dims[2];
        final int embeddingSz = x.dims[1];


        // y is the deltas
        // y.dims[0] is the #feature maps in the output
        final int nFeatureMapsOutput = ygrad.dims[0];
        // y.dims[1] is the #rows in output
        final int yRows = ygrad.dims[2];

        // The number of cubes is the output depth (# feature maps)
        Tensor weightGrads = new Tensor(nFeatureMapsOutput, nFeatureMapsInput, embeddingSz, xRows - yRows + 1);

        final int kRows = weightGrads.dims[3];

        // For each feature map
        for (int k = 0; k < nFeatureMapsInput; ++k)
        {
            // The weight gradient is the size of the kernel itself, which is a cube of size input depth x kh x kw
            for (int l = 0; l < nFeatureMapsOutput; ++l)
            {
                // For each input col, embeddingSz is also kCols essentially
                for (int j = 0; j < embeddingSz; ++j)
                {
                    // For each input row
                    for (int i = 0; i < kRows; ++i)
                    {

                        int wAddr = ((l * nFeatureMapsInput + k) * embeddingSz + j) * kRows + i;
                        // For input depth
                        double acc = 0;

                        // corr2!!
                        for (int m = 0; m < yRows; ++m)
                        {

                            int xAddr = (k * embeddingSz + j) * xRows + i + m;
                            int yAddr = (l * embeddingSz + j) * yRows + m;
                            acc += x.d[xAddr] * ygrad.d[yAddr];
                        }
                        weightGrads.d[wAddr] = acc;
                    }
                }
            }
        }
        return weightGrads;
    }

    public static Tensor corr1MM(Tensor data, Tensor kernels, double[] biases)
    {
        final int iT = data.dims[1];
        final int embedSz = data.dims[2];
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kW = kernels.dims[2];
        final int oT = iT - kW + 1;
        Tensor output = new Tensor(nK, oT, embedSz);
        // We are going to make a dataM[l] matrix of size (oT, kW * kL)


        double[] dataM = new double[oT * kW * kL];
        double[] outputM = new double[nK * oT * embedSz];
        // Make an unfolded filter that is (nL * kW, nK)
        double[] filter = new double[nK * kL * kW];

        // Makes the unfolded data matrix


        for (int ei = 0; ei < embedSz; ++ei)
        {

            int zk = 0;

            for (int k = 0; k < nK; ++k)
            {

                for (int l = 0; l < kL; ++l)
                {
                    for (int a = 0; a < kW; ++a)
                    {
                        filter[zk++] = kernels.d[((k * kL + l) * kW + a) * embedSz + ei];
                    }
                }
            }

            int z = 0;


            for (int k = 0; k < kL; ++k)
            {

                for (int j = 0; j < kW; ++j)
                {
                    for (int i = 0; i < oT; ++i)
                    {

                        // k * oT is the offset for the kth kernel
                        dataM[z++] = data.d[(k * iT + i + j) * embedSz + ei];
                    }
                }
            }

            NativeBlas.dgemm('N', 'N', oT, nK, kW * kL, 1.0, dataM, 0, oT, filter, 0, kW * kL, 0, outputM, 0, oT);

            int zo = 0;
            for (int k = 0; k < nK; ++k)
            {
                double bias = 0;
                if (biases != null)
                {
                    bias = biases[k];
                }
                for (int o = 0; o < oT; ++o)
                {
                    output.d[(k * oT + o) * embedSz + ei] = outputM[zo++] + bias;

                }
            }
        }
        // Make the unfolded kernel matrix
        return output;

    }


    public static Tensor conv1MM(Tensor data, Tensor kernels, double[] biases)
    {
        final int iT = data.dims[1];
        final int embedSz = data.dims[2];
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kW = kernels.dims[2];
        final int oT = iT - kW + 1;
        Tensor output = new Tensor(nK, oT, embedSz);
        // We are going to make a dataM[l] matrix of size (oT, kW * kL)


        double[] dataM = new double[oT * kW * kL];
        double[] outputM = new double[nK * oT * embedSz];
        // Make an unfolded filter that is (nL * kW, nK)
        double[] filter = new double[nK * kL * kW];

        // Makes the unfolded data matrix


        for (int ei = 0; ei < embedSz; ++ei)
        {

            int zk = 0;

            for (int k = 0; k < nK; ++k)
            {

                for (int l = 0; l < kL; ++l)
                {
                    for (int a = kW - 1; a >= 0; --a)
                    {
                        filter[zk++] = kernels.d[((k * kL + l) * kW + a) * embedSz + ei];
                    }
                }
            }

            int z = 0;


            for (int k = 0; k < kL; ++k)
            {

                for (int j = 0; j < kW; ++j)
                {
                    for (int i = 0; i < oT; ++i)
                    {

                        // k * oT is the offset for the kth kernel
                        dataM[z++] = data.d[(k * iT + i + j) * embedSz + ei];
                    }
                }
            }

            NativeBlas.dgemm('N', 'N', oT, nK, kW * kL, 1.0, dataM, 0, oT, filter, 0, kW * kL, 0, outputM, 0, oT);

            int zo = 0;
            for (int k = 0; k < nK; ++k)
            {
                double bias = 0;
                if (biases != null)
                {
                    bias = biases[k];
                }
                for (int o = 0; o < oT; ++o)
                {
                    output.d[(k * oT + o) * embedSz + ei] = outputM[zo++] + bias;

                }
            }
        }
        // Make the unfolded kernel matrix
        return output;

    }

    public static Tensor fftfilt(FFTOps fft, Tensor data, Tensor kernels, double[] biases, boolean corr)
    {
        final int iT = data.dims[2];
        final int embedSz = data.dims[1];
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kW = kernels.dims[3];
        final int oT = iT - kW + 1;
        Tensor output = new Tensor(nK, embedSz, oT);
        double[] z = new double[oT];
        for (int k = 0; k < nK; ++k)
        {
            final double bias = biases == null ? 0.0 : biases[k];

            for (int j = 0, dOff = 0; j < embedSz; ++j, dOff += iT)
            {
                for (int l = 0; l < kL; ++l)
                {
                    int dataAddr0 = (l * embedSz + j) * iT;
                    int kernAddr0 = ((k * kL + l) * embedSz + j) * kW;
                    int outAddr0 = (k * embedSz + j) * oT;

                    fft.filter(data.d, dataAddr0, iT, kernels.d, kernAddr0, kW, z, corr);
                    for (int i = 0; i < oT; ++i)
                    {
                        output.d[outAddr0 + i] += z[i] + bias;
                    }
                }
            }
        }
        return output;
    }

    public static Tensor corr1(Tensor data, Tensor kernels, double[] biases)
    {
        final int iT = data.dims[2];
        final int embedSz = data.dims[1];
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kW = kernels.dims[3];
        final int oT = iT - kW + 1;
        Tensor output = new Tensor(nK, embedSz, oT);

        for (int k = 0; k < nK; ++k)
        {
            final double bias = biases == null ? 0.0 : biases[k];

            for (int j = 0; j < embedSz; ++j)
            {
                final int outAddr0 = (k * embedSz + j) * oT;

                for (int i = 0; i < oT; ++i)
                {
                    output.d[outAddr0 + i] = bias;
                }

                for (int l = 0; l < kL; ++l)
                {
                    final int dataAddr0 = (l * embedSz + j) * iT;
                    final int kernAddr0 = ((k * kL + l) * embedSz + j) * kW;

                    for (int i = 0; i < oT; ++i)
                    {
                        final int outAddr = outAddr0 + i;
                        for (int m = 0; m < kW; ++m)
                        {
                            final int dataAddr = dataAddr0 + i + m;
                            final int kernAddr = kernAddr0 + m;
                            output.d[outAddr] += data.d[dataAddr] * kernels.d[kernAddr];
                        }
                    }
                }
            }
        }
        return output;
    }


    public static Tensor conv1(Tensor data, Tensor kernels, double[] biases)
    {
        final int iT = data.dims[2];
        final int embedSz = data.dims[1];
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        // Note that this is still the 3rd argument.  Now the fourth arg is not used
        final int kW = kernels.dims[3];
        final int oT = iT - kW + 1;
        Tensor output = new Tensor(nK, embedSz, oT);

        for (int k = 0; k < nK; ++k)
        {
            final double bias = biases == null ? 0.0 : biases[k];

            for (int j = 0; j < embedSz; ++j)
            {
                final int outAddr0 = (k * embedSz + j) * oT;

                for (int i = 0; i < oT; ++i)
                {
                    output.d[outAddr0 + i] = bias;
                }

                for (int l = 0; l < kL; ++l)
                {

                    final int dataAddr0 = (l * embedSz + j) * iT;
                    final int kernAddr0 = ((k * kL + l) * embedSz + j) * kW;


                    for (int i = 0; i < oT; ++i)
                    {
                        final int outAddr = outAddr0 + i;
                        for (int m = 0; m < kW; ++m)
                        {
                            final int dataAddr = dataAddr0 + i + m;
                            final int kernAddr = kernAddr0 + (kW - m - 1);
                            output.d[outAddr] += data.d[dataAddr] * kernels.d[kernAddr];
                        }
                    }
                }
            }
        }
        return output;
    }

    // In this case, we have several feature maps, each is a kernel of

    public static Tensor conv2(Tensor data, Tensor kernels, double[] biases)
    {
        final int dH = data.dims[1];
        final int dW = data.dims[2];
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kH = kernels.dims[2];
        final int kW = kernels.dims[3];
        final int oH = dH - kH + 1;
        final int oW = dW - kW + 1;
        Tensor output = new Tensor(nK, oH, oW);

        for (int k = 0; k < nK; ++k)
        {
            int kbase = k * kL;
            int obase = k * oH;

            final double bias = biases == null ? 0.0 : biases[k];
            for (int i = 0; i < oH; ++i)
            {
                int ibase = (obase + i) * oW;
                for (int j = 0; j < oW; ++j)
                {
                    int outAddr = ibase + j;
                    double acc = 0.;
                    for (int l = 0; l < kL; ++l)
                    {
                        for (int m = 0; m < kH; ++m)
                        {
                            for (int n = 0; n < kW; ++n)
                            {
                                int dataAddr = (l * dH + i + m) * dW + j + n;
                                int mh = kH - m - 1;
                                int nw = kW - n - 1;
                                int kernAddr = ((kbase + l) * kH + mh) * kW + nw;
                                acc += data.d[dataAddr] * kernels.d[kernAddr];
                            }
                        }
                    }
                    output.d[outAddr] = acc + bias;
                }
            }
        }
        return output;
    }

    public static Tensor corr2(Tensor data, Tensor kernels, double[] biases)
    {
        final int dH = data.dims[1];
        final int dW = data.dims[2];
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kH = kernels.dims[2];
        final int kW = kernels.dims[3];
        final int oH = dH - kH + 1;
        final int oW = dW - kW + 1;
        Tensor output = new Tensor(nK, oH, oW);

        for (int k = 0; k < nK; ++k)
        {
            int kbase = k * kL;
            int obase = k * oH;

            final double bias = biases == null ? 0.0 : biases[k];
            for (int i = 0; i < oH; ++i)
            {
                int ibase = (obase + i) * oW;
                for (int j = 0; j < oW; ++j)
                {
                    int outAddr = ibase + j;
                    double acc = 0.;
                    for (int l = 0; l < kL; ++l)
                    {
                        for (int m = 0; m < kH; ++m)
                        {
                            for (int n = 0; n < kW; ++n)
                            {
                                int dataAddr = (l * dH + i + m) * dW + j + n;
                                int kernAddr = ((kbase + l) * kH + m) * kW + n;
                                double d = data.d[dataAddr] * kernels.d[kernAddr];
                                acc += d;

                            }
                        }
                    }
                    output.d[outAddr] = acc + bias;

                }
            }
        }
        return output;

    }
}
/*
    // Here we do like Chellapilla et. al. and conv2 by collapsing
    // The input matrix gets transformed row by row to be the full h*w of the kernel
    // concatenated by each band at each lag.  So the height is the number of lags, which is the same as the output
    // dimension
    //
    // X = [ Ctr1; Ctr2; Ctr3; .... NConvs ]
    // Lets say the
    // CtrX is then the kernel width * kernel height.
    // K = [f1k1 . f1k2 . f1k3; f2k1 . f2k2 . f3k3; ... Num features ];
    // Y = [fm1; fm2]
    //
    // Forward prop: Y = X * W
    //
    // Y = | Conv1 |
    //     | Conv2 |
    //     | Conv3 |
    //
    // Back prop: gradX = gradY * trans(W)
    //
    // Weight grad: gradW = trans(X) * gradY
    // Correlation between Feature Map 1, ... N and each window in the kernel
    // This shape looks exactly like the kernel matrix, where each column is a kernel and each row is an output feature map
    //
    //
    public static Tensor conv2MM(Tensor data, Tensor kernels, boolean flip)
    {
        // kernels.dims[0] = kernel
        // kernels.dims[1] = layer
        // kernels.dims[2] = height
        // kernels.dims[3] = width
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kH = kernels.dims[2];
        final int kW = kernels.dims[3];
        final int dL = data.dims[0];
        final int dH = data.dims[1];
        final int dW = data.dims[2];
        final int singleWidth = kH * kW;
        final int cubeW = dL * singleWidth;
        final int cubeH = (dH - kH + 1) * (cubeW - kW + 1);
        Rectangle input = new Rectangle(cubeH, cubeW);
        Rectangle filter = new Rectangle(kH * kW * kL, nK);
        Tensor output = new Tensor(nK, data.dims[1] - kH + 1, data.dims[2] - kW + 1);

        // To populate the input, we need a loop over the full kernel width
        int n = 0;

        for (int k = 0; k < dL; ++k)
        {
            for (int m = 0; m < kH; ++m)
            {
                for (int p = 0; p < kW; ++p)
                {
                    for (int h = 0; h < dH - kH + 1; ++h)
                    {
                        for (int w = 0; w < dW - kW + 1; ++w)
                        {
                            int inAddr = (k * dH + h + m) * dW + w + p;

                            double d = data.d[inAddr];
                            input.d[n] = d;
                            ++n;
                        }
                    }
                }
            }
        }

        if (!flip)
        {
            unrollKernel2(kernels, filter);
        } else
        {
            unrollFlipKernel2(kernels, filter);
        }
        NativeBlas.dgemm('N', 'N', input.h, filter.w, input.w, 1.0, input.d, 0, input.h, filter.d, 0, filter.h, 0, output.d, 0, input.h);
        return output;
    }

    private static void unrollKernel2(Tensor kernels, Rectangle filter)
    {
        int n;
        n = 0;
        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kH = kernels.dims[2];
        final int kW = kernels.dims[3];
        for (int k = 0; k < nK; ++k)
        {
            int kbase = k * kL;

            for (int l = 0; l < kL; ++l)
            {
                int lbase = (kbase + l) * kH;
                for (int a = 0; a < kH; ++a)
                {
                    int abase = (lbase + a) * kW;
                    for (int b = 0; b < kW; ++b)
                    {
                        int idx = abase + b;
                        filter.d[n] = kernels.d[idx];
                        ++n;
                    }
                }
            }
        }
    }

    private static void unrollFlipKernel2(Tensor kernels, Rectangle filter)
    {
        int n;
        n = 0;

        final int nK = kernels.dims[0];
        final int kL = kernels.dims[1];
        final int kH = kernels.dims[2];
        final int kW = kernels.dims[3];
        for (int k = 0; k < nK; ++k)
        {
            int kbase = k * kL;

            for (int l = 0; l < kL; ++l)
            {
                int lbase = (kbase + l) * kH;
                for (int a = 0; a < kH; ++a)
                {
                    int abase = (lbase + kH - a - 1) * kW;
                    for (int b = 0; b < kW; ++b)
                    {
                        int idx = abase + kW - b - 1;
                        filter.d[n] = kernels.d[idx];
                        ++n;
                    }
                }
            }
        }
    }
}
*/