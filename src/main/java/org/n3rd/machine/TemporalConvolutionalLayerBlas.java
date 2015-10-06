package org.n3rd.machine;


import org.jblas.NativeBlas;
import org.n3rd.Tensor;
import org.n3rd.layers.AbstractLayer;

import java.util.ArrayList;
import java.util.LinkedHashMap;

/**
 * k = 1:len(outputFeatureMaps)
 * j = 1:len(inputFeatureMaps)
 * i = 1:len(embedSz)
 * [ a_ji b_ji c_ji d_ji e_ji f_ji g_ji ] o [ z_kji y_kji x_kji ]
 *
 * No Embeddings is pretty simple:
 *
 * || a_00 b_00 c_00 | a_10 b_10 c_10 |      || z_000 | z_100 ||
 * || b_00 c_00 d_00 | b_10 c_10 d_10 |      || y_000 | y_100 ||
 * || c_00 d_00 e_00 | c_10 d_10 e_10 |      || x_000 | x_100 ||
 * || d_00 e_00 f_00 | d_10 e_10 f_10 |      || z_010 | z_110 ||
 *                                           || y_010 | y_110 ||
 *                                           || x_010 | x_110 ||
 *
 *
 * With embeddings we end up having another set of columns, and some zero filters.
 * K = num taps, N is length series, output matrix is  (N - K + 1, oFsz * embedSz)
 *
 *      IFM1e0            IFM2e0            IFM1e1          IFM2e1
 *
 * || a_00 b_00 c_00 | a_10 b_10 c_10 | a_01 b_01 c_01 | a_11 b_11 c_11 |      || z_000 | 0     | z_100 | 0     ||
 * || b_00 c_00 d_00 | b_10 c_10 d_10 | b_01 c_01 d_01 | b_11 c_11 d_11 |      || y_000 | 0     | y_100 | 0     ||
 * || c_00 d_00 e_00 | c_10 d_10 e_10 | c_01 d_01 e_01 | c_11 d_11 e_11 |      || x_000 | 0     | x_100 | 0     ||
 * || d_00 e_00 f_00 | d_10 e_10 f_10 | d_01 e_01 f_01 | d_11 e_11 f_11 |      || z_010 | 0     | z_110 | 0     ||
 *                                                                             || y_010 | 0     | y_110 | 0     ||
 *                                                                             || x_010 | 0     | x_110 | 0     ||
 *                                                                             || 0     | z_001 | 0     | z_101 ||
 *                                                                             || 0     | y_001 | 0     | y_101 ||
 *                                                                             || 0     | x_001 | 0     | x_101 ||
 *                                                                             || 0     | z_011 | 0     | z_111 ||
 *                                                                             || 0     | y_011 | 0     | y_111 ||
 *                                                                             || 0     | x_011 | 0     | x_111 ||
 *        OFM1     |      OFM2
 * --------------------------------
 * || ofm1e00 ofm1e10 ofm2e00 ofm2e10
 * || ofm1e01 ofm1e11 ofm2e01 ofm2e11
 * || ofm1e02 ofm1e12 ofm2e02 ofm2e12
 * || ofm1e03 ofm1e13 ofm2e03 ofm2e13
 *
 */
public class TemporalConvolutionalLayerBlas extends AbstractLayer
{


    int nK;
    int kL;
    int embedSz;
    int numFrames;
    int kW;

    // Input is Number of frames x frame width (num feature maps)
    Tensor unwrappedInput;

    public TemporalConvolutionalLayerBlas()
    {

    }

    public TemporalConvolutionalLayerBlas(int nK, int kL, int kW, int embeddingSize)
    {

        this.nK = nK;
        this.kL = kL;
        this.kW = kW;
        this.embedSz = embeddingSize;

        weights = new Tensor(kL * embeddingSize * kW, nK * embeddingSize);
        gradsW = new Tensor(kL * embeddingSize * kW, nK * embeddingSize);
        biases = new double[nK];
        biasGrads = new double[nK];


        int pitch = weights.dims[0];
        // For each kernel, randomly initialize all weights
        for (int j = 0; j < embedSz; ++j)
        {

            for (int i = 0; i < kL; ++i)
            {
                for (int k = 0; k < nK; ++k)
                {
                    for (int m = 0; m < kW; ++m)
                    {
                        // we need to get eSz * j + i row and eSz * j + k col
                        //
                        int row = (j * kL + i) * kW + m;
                        int col = k * embedSz + j;
                        int addr =  col * pitch + row;
                        weights.d[addr] = rand();
                    }
                }

            }
        }
    }

    public double rand()
    {

        //final int embeddingSz = weights.dims[3];
        double stdv = 1. / Math.sqrt(6. / 28.);
        double stdv2 = stdv * 2;
        double d = Math.random() * stdv2 - stdv;
        return d;
    }

    private static void reorderOutput(Tensor unwrapped, int nK, int embedSz)
    {
        //Tensor output = new Tensor(oT, nK * embedSz);
        // We have committed to unwrapping our output matrix to the form
        int oT = unwrapped.dims[0];
        unwrapped.dims = new int[] {nK,oT,embedSz};
        double[] out = new double[nK*oT*embedSz];
        // This looks like
        // output[k][i][j] ==
        // (k * oT + i) * eSz + j
        // Our current data is laid out in fortran row major form as follows
        //
        // unwrapped[i][k][j] ==
        // (k * eSz + j) * oT + i

        // You could also do nIdx++ I think
        for (int k = 0; k < nK; ++k)
        {
            for (int i = 0; i < oT; ++i)
            {
                for (int j = 0; j < embedSz; ++j)
                {
                    int nIdx = (k * oT + i) * embedSz + j;
                    int cIdx = (k * embedSz + j) * oT + i;
                    //int cIdx = (j * nK + k) * oT + i;
                    double old = unwrapped.d[cIdx];
                    //unwrapped.d[cIdx] = unwrapped.d[nIdx];
                    //unwrapped.d[nIdx] = old;
                    out[nIdx] = old;
                }
            }
        }
        unwrapped.d = out;
    }

    private static Tensor unwrapGrad(Tensor chainGrad, int nK, int embedSz)
    {
        final int oT = chainGrad.dims[1];
        Tensor unwrapped = new Tensor(oT, nK * embedSz);

        // You could also do nIdx++ I think
        for (int k = 0; k < nK; ++k)
        {
            for (int i = 0; i < oT; ++i)
            {
                for (int j = 0; j < embedSz; ++j)
                {
                    int nIdx = (k * oT + i) * embedSz + j;
                    int cIdx = (k * embedSz + j) * oT + i;

                    unwrapped.d[cIdx] = chainGrad.d[nIdx];
                }
            }
        }
        return unwrapped;
    }

    private static Tensor unwrapX(Tensor x, int kW)
    {
        final int kL = x.dims[0];
        final int iT = x.dims[1];
        final int embedSz = x.dims[2];
        final int oT = iT - kW + 1;
        Tensor unwrapped = new Tensor(oT, kW * kL * embedSz);

        // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
        int n = 0;
        for (int j = 0; j < embedSz; ++j)
        {
            for (int k = 0; k < kL; ++k)
            {
                for (int m = 0; m < kW; ++m)
                {
                    for (int i = 0; i < oT; ++i)
                    {
                        int offset = (k * iT + i + m) * embedSz + j;
                        // x(kL, iT, embedSz)
                        unwrapped.d[n++] = x.d[offset];
                    }
                }
            }
        }
        return unwrapped;
    }

    private void wrapX(Tensor unwrapped, Tensor grads, int kW)
    {

        final int oT = unwrapped.dims[0];
        final int iT = oT + kW - 1;
        assert(iT == grads.dims[1]);
        final int kL = grads.dims[0];
        final int embedSz = grads.dims[2];
        assert(unwrapped.dims[1] / kW / kL == embedSz);


        // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
        int n = 0;
        for (int j = 0; j < embedSz; ++j)
        {
            for (int k = 0; k < kL; ++k)
            {
                for (int m = 0; m < kW; ++m)
                {
                    for (int i = 0; i < oT; ++i)
                    {
                        int offset = (k * iT + i + m) * embedSz + j;
                        // x(kL, iT, embedSz)
                         grads.d[offset] += unwrapped.d[n++];
                    }
                }
            }
        }

    }

    @Override
    public Tensor forward(Tensor z)
    {
        // For convolutions, we should assume that our VectorN is truly a matrix
        // and the usual math applies

        numFrames = z.size() / embedSz / kL;
        Tensor input = new Tensor(z.d, kL, numFrames, embedSz);
        grads = new Tensor(kL, numFrames, embedSz);

        final int oT = numFrames - kW + 1;
        output = new Tensor(oT, nK * embedSz);
        unwrappedInput = unwrapX(input, kW);

        NativeBlas.dgemm('N', 'N', unwrappedInput.dims[0], weights.dims[1], unwrappedInput.dims[1], 1.0, unwrappedInput.d, 0, unwrappedInput.dims[0],
                weights.d, 0, weights.dims[0], 0, output.d, 0, output.dims[0]);

        reorderOutput(output, nK, embedSz);
        return output;
    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {


        final int oT = numFrames - kW + 1;
        int[] outputDims = new int[] { nK, oT, embedSz };
        Tensor chainGradTensor = new Tensor(chainGrad.d, outputDims);
        Tensor unwrappedChainGrad = unwrapGrad(chainGradTensor, nK, embedSz);
        Tensor unwrappedGradInput = new Tensor(oT, kW * kL * embedSz);

        int m = unwrappedChainGrad.dims[0];
        int k = unwrappedChainGrad.dims[1];
        int n = weights.dims[0];

        NativeBlas.dgemm('N', 'T', m, n, k, 1.0, unwrappedChainGrad.d, 0, m,
                weights.d, 0, n, 0, unwrappedGradInput.d, 0, m);

        m = unwrappedInput.dims[1];
        k = unwrappedInput.dims[0];
        n = unwrappedChainGrad.dims[1];

        NativeBlas.dgemm('T', 'N', m, n, k, 1.0, unwrappedInput.d, 0, k,
                unwrappedChainGrad.d, 0, k, 0, gradsW.d, 0, m);

        // Because of the way we unrolled the embeddings matrix, we actually are allowing the previous computation
        // to calculate values that can't and should'nt be applied to the weight
        for (int i = 0, sz = weights.size(); i < sz; ++i)
        {
            if (weights.d[i] == 0.0)
            {
                gradsW.d[i] = 0.;
            }
        }

        // We need to update gradsW, which are (kL * embeddingSize) * kW x (nK * embeddingSize);
        /*
        int stride = convOutputSz * embedSz;
        for (int l = 0; l < nK; ++l)
        {
            for (int i = 0; i < stride; ++i)
            {
                this.biasGrads[l] += chainGradX[l * stride + i];
            }
            this.biasGrads[l] /= embedSz;
        }*/


        wrapX(unwrappedGradInput, grads, kW);

        return grads;
    }


    /**
     * Exists for reserialization purposes only!  Turns instantly into a Tensor when injected
     *
     * @param params
     */
    public void setParams(LinkedHashMap<String, Object> params)
    {
        this.weights = new Tensor(params);
    }

    public void setBiasParams(ArrayList<Double> biasParams)
    {
        int sz = biasParams.size();
        biases = new double[sz];
        for (int i = 0; i < sz; ++i)
        {
            biases[i] = biasParams.get(i);
        }
    }

}
