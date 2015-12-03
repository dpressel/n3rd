package org.n3rd.layers;

import org.jblas.NativeBlas;
import org.n3rd.Tensor;
import org.sgdtk.ArrayDouble;
import org.sgdtk.DenseVectorN;
import org.sgdtk.VectorN;

import java.util.ArrayList;
import java.util.LinkedHashMap;

/**
 * k = 1:len(outputFeatureMaps)
 * j = 1:len(inputFeatureMaps)
 * i = 1:len(embedSz)
 * [ a_ji b_ji c_ji d_ji e_ji f_ji g_ji ] o [ z_kji y_kji x_kji ]
 * <p/>
 * No Embeddings is pretty simple:
 * <p/>
 * || a_00 b_00 c_00 | a_10 b_10 c_10 |      || z_000 | z_100 ||
 * || b_00 c_00 d_00 | b_10 c_10 d_10 |      || y_000 | y_100 ||
 * || c_00 d_00 e_00 | c_10 d_10 e_10 |      || x_000 | x_100 ||
 * || d_00 e_00 f_00 | d_10 e_10 f_10 |      || z_010 | z_110 ||
 * || y_010 | y_110 ||
 * || x_010 | x_110 ||
 * <p/>
 * *        OFM1
 * ----------------
 * || ofm1e00 ofm2e00
 * || ofm1e01 ofm2e01
 * || ofm1e02 ofm2e02
 * || ofm1e03 ofm2e03
 */
public class TemporalConvolutionalLayerBlas implements Layer
{

    Tensor gradsW;
    Tensor grads;
    double[] biases;
    double[] biasGrads;

    int nK;

    @Override
    public Tensor getOutput()
    {
        return output;
    }

    int kL;
    int numFrames;
    int kW;

    Tensor weights;

    // Input is Number of frames x frame width (num feature maps)
    Tensor unwrappedInput;
    Tensor output;

    public TemporalConvolutionalLayerBlas()
    {
    }

    public TemporalConvolutionalLayerBlas(int nK, int kL, int kW)
    {

        this.nK = nK;
        this.kL = kL;
        this.kW = kW;

        output = new Tensor(1);
        grads = new Tensor(1);
        unwrappedInput = new Tensor(1);
        weights = new Tensor(kL * kW, nK);
        gradsW = new Tensor(kL * kW, nK);
        biases = new double[nK];
        biasGrads = new double[nK];

        for (int i = 0; i < weights.size(); ++i)
        {
            weights.set(i, rand());
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

    // Straight transpose on the output
    private void reorderOutput(ArrayDouble unwrapped)
    {
        //Tensor output = new Tensor(oT, nK);
        // We have committed to unwrapping our output matrix to the form
        int oT = numFrames - kW + 1;


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

                int nIdx = k * oT + i;
                int cIdx = i * nK + k;
                double tmp = unwrapped.get(nIdx);
                unwrapped.set(nIdx, unwrapped.get(cIdx));
                unwrapped.set(cIdx, tmp);
            }
        }

    }

    // When going backwards we need a simple transpose, but here we make a copy so as not
    // to destroy the other chain value
    private Tensor unwrapGradFromNextLayer(Tensor chainGrad, Tensor unwrapped)
    {
        final int oT = numFrames - kW + 1;

        // You could also do nIdx++ I think
        for (int k = 0; k < nK; ++k)
        {
            for (int i = 0; i < oT; ++i)
            {
                int nIdx = k * oT + i;
                int cIdx = i * nK + k;

                unwrapped.set(nIdx, chainGrad.get(cIdx));
            }
        }
        return unwrapped;
    }

    private void unwrapInput(ArrayDouble x)
    {

        final int oT = numFrames - kW + 1;
        unwrappedInput.resize(oT, kW * kL);
        int n = 0;


        for (int k = 0; k < kL; ++k)
        {

            for (int m = 0; m < kW; ++m)
            {
                for (int i = 0; i < oT; ++i)
                {

                    int offset = k * numFrames + i + m;
                    unwrappedInput.set(n, x.at(offset));
                    ++n;
                }
            }
        }

    }

    private void wrapGrad(Tensor unwrapped)
    {

        final int oT = unwrapped.dims[0];
        final int iT = oT + kW - 1;
        assert (iT == grads.dims[2]);
        final int kL = grads.dims[0];
        final int embedSz = grads.dims[1];
        assert (embedSz == 1);
        // grads = {oT, kW * kL}

        // In the blas case, we need to write in column major, which means write down one lag, then move up to the next
        int n = 0;
        for (int k = 0; k < kL; ++k)
        {
            for (int m = 0; m < kW; ++m)
            {
                for (int i = 0; i < oT; ++i)
                {
                    int offset = k * iT + i + m;
                    grads.addi(offset, unwrapped.get(n));
                    n++;
                }

            }
        }



    }

    @Override
    public Tensor forward(Tensor z)
    {

        numFrames = z.size() / kL;
        grads.resize(kL, 1, numFrames);
        grads.constant(0.);

        final int oT = numFrames - kW + 1;

        output.resize(nK, 1, oT);

        ArrayDouble input = z.getArray();

        unwrapInput(input);

        output.constant(0.);
        NativeBlas.dgemm('N', 'N', unwrappedInput.dims[0], weights.dims[1], unwrappedInput.dims[1], 1.0,
                unwrappedInput.getArray().v, 0, unwrappedInput.dims[0],
                weights.getArray().v, 0, weights.dims[0], 0, output.getArray().v, 0, oT);

        reorderOutput(output.getArray());

        return output;

    }

    @Override
    public Tensor backward(Tensor chainGrad, double y)
    {

        try
        {

            final int oT = numFrames - kW + 1;
            int[] outputDims = new int[]{nK, 1, oT};
            chainGrad.reshape(outputDims);
            Tensor unwrappedChainGrad = new Tensor(oT, nK);
            unwrapGradFromNextLayer(chainGrad, unwrappedChainGrad);
            Tensor unwrappedGradInput = new Tensor(oT, kW * kL);
            int m = unwrappedChainGrad.dims[0];
            int k = unwrappedChainGrad.dims[1];
            int n = weights.dims[0];

            NativeBlas.dgemm('N', 'T', m, n, k, 1.0, unwrappedChainGrad.getArray().v, 0, m,
                    weights.getArray().v, 0, n, 0, unwrappedGradInput.getArray().v, 0, m);

            m = unwrappedInput.dims[1];
            k = unwrappedInput.dims[0];
            n = unwrappedChainGrad.dims[1];

            NativeBlas.dgemm('T', 'N', m, n, k, 1.0, unwrappedInput.getArray().v, 0, k,
                    unwrappedChainGrad.getArray().v, 0, k, 0, gradsW.getArray().v, 0, m);

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


            wrapGrad(unwrappedGradInput);

            return grads;
        }
        catch (Exception ex)
        {
            throw new RuntimeException(ex);
        }
    }

    @Override
    public Tensor getParamGrads()
    {
        return gradsW;
    }

    @Override
    public Tensor getParams()
    {
        return weights;
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

    @Override
    public double[] getBiasGrads()
    {
        return biasGrads;
    }

    @Override
    public double[] getBiasParams()
    {
        return biases;
    }

}
