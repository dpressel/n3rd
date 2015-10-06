package org.n3rd;

import java.util.List;
import java.util.Map;

/**
 * Most basic implementation of a Tensor ever -- limited to exactly what is being used at the moment
 *
 * @author dpressel
 */
public class Tensor
{
    public Tensor(Map<String, Object> map)
    {
        List<Integer> dimList = (List<Integer>)map.get("dims");
        dims = new int[dimList.size()];
        int length = 1;
        for (int i = 0; i < dims.length; ++i)
        {
            dims[i] = dimList.get(i);
            length *= dims[i];
        }
        sz = length;
        List<Double> values = (List<Double>)map.get("d");
        d = new double[values.size()];
        for (int i = 0; i < d.length; ++i)
        {
            d[i] = values.get(i);
        }
    }
    public Tensor(int... dims)
    {
        this.dims = new int[dims.length];

        int length = 1;
        for (int i = 0; i < dims.length; ++i)
        {
            this.dims[i] = dims[i];
            length *= dims[i];
        }
        sz = length;
        d = new double[length];
    }

    public Tensor(double[] x, int... dims)
    {
        this.dims = new int[dims.length];
        int length = 1;
        for (int i = 0; i < dims.length; ++i)
        {
            this.dims[i] = dims[i];
            length *= dims[i];
        }

        sz = length;
        d = x;
    }
    public Tensor(Tensor x)
    {
        sz = x.size();
        this.dims = new int[x.dims.length];

        for (int i = 0; i < x.dims.length; ++i)
        {
            this.dims[i] = x.dims[i];

        }
        d = new double[sz];
        System.arraycopy(x.d, 0, d, 0, x.size());
    }

    public void reshape(int... dims) throws Exception
    {
        this.dims = new int[dims.length];

        int length = 1;
        for (int i = 0; i < dims.length; ++i)
        {
            this.dims[i] = dims[i];
            length *= dims[i];
        }

        if (length != size())
        {
            throw new Exception("Lengths must agree for reshape");
        }
    }

    public void resize(int... dims)
    {
        this.dims = new int[dims.length];

        int length = 1;
        for (int i = 0; i < dims.length; ++i)
        {
            this.dims[i] = dims[i];
            length *= dims[i];
        }



        if (size() != length)
        {
            sz = length;
            double [] old = d;
            d = new double[length];

            int minLength = Math.min(old.length, d.length);
            System.arraycopy(old, 0, d, 0, minLength);
        }
    }


    public int[] dims;
    public int sz;
    public double[] d;

    public int size()
    {
        return sz;
    }
    public void reset(double x)
    {

        for (int i = 0; i < sz; ++i)
        {
            d[i] = x;
        }
    }

    public static Tensor replicate(Tensor tensor, int times)
    {
        int[] newDims = new int[tensor.dims.length];
        for (int i = 0; i < tensor.dims.length; ++i)
        {
            newDims[i] = tensor.dims[i];
        }
        newDims[0] *= times;

        Tensor replicated = new Tensor(newDims);
        int pitch = tensor.size();
        for (int i = 0; i < pitch; ++i)
        {
            for (int k = 0; k < newDims[0]; ++k)
            {
                replicated.d[k * pitch + i] = tensor.d[i];
            }
        }
        return replicated;
    }

    public static Tensor transposeWeight4D(Tensor weight)
    {

        int[] newDims = new int[weight.dims.length];

        newDims[0] = weight.dims[1];
        newDims[1] = weight.dims[0];
        newDims[2] = weight.dims[2];
        newDims[3] = weight.dims[3];
        // If either feature map is one, no need to copy, memory is the same
        if (weight.dims[0] == 1 || weight.dims[1] == 1)
        {
            return new Tensor(weight.d, newDims);
        }
        Tensor weightCopy = new Tensor(newDims);
        int sz = newDims[2] * newDims[3];
        for (int i = 0; i < newDims[0]; ++i)
        {

            for (int j = 0; j < newDims[1]; ++j)
            {
                for (int k = 0; k < sz; ++k)
                {
                    weightCopy.d[(i * newDims[1] + j) * sz + k] = weight.d[(j * weight.dims[1] + i) * sz + k];
                }
            }
        }
        return weightCopy;

    }

    public static Tensor embed(Tensor tensor, int h, int w)
    {
        return embed(tensor, 0, h, w);
    }
    public static Tensor embed(Tensor tensor, int l, int h, int w)
    {
        final int tL = tensor.dims[0];
        final int tH = tensor.dims[1];
        final int tW = tensor.dims[2];
        final int oH = tH + h;
        final int oW = tW + w;
        final int oL = tL + l;
        Tensor zpCube = new Tensor(oL, oH, oW);
        int lStart = l/2;
        int hStart = h/2;
        int wStart = w/2;
        for (int k = 0; k < tL; ++k)
        {
            for (int i = 0; i < tH; ++i)
            {
                for (int j = 0; j < tW; ++j)
                {
                    zpCube.d[((k + lStart) * oH + i + hStart) * oW + j + wStart] = tensor.d[(k * tH + i) * tW + j];
                }
            }
        }
        return zpCube;
    }
}


