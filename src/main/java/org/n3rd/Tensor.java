package org.n3rd;

import org.sgdtk.ArrayDouble;

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
        array = new ArrayDouble(length);

        List<Double> values = (List<Double>)map.get("d");


        for (int i = 0; i < length; ++i)
        {
            array.set(i, values.get(i));
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
        array = new ArrayDouble(length);


    }


    public Tensor(ArrayDouble x, int... dims)
    {
        this.dims = new int[dims.length];
        int length = 1;
        for (int i = 0; i < dims.length; ++i)
        {
            this.dims[i] = dims[i];
            length *= dims[i];
        }
        assert(length == x.size());
        array = new ArrayDouble(x);

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
        array = new ArrayDouble(x);

    }



    public Tensor(Tensor x)
    {

        this.dims = new int[x.dims.length];

        for (int i = 0; i < x.dims.length; ++i)
        {
            this.dims[i] = x.dims[i];

        }
        array = new ArrayDouble(x.getArray());
    }

    private ArrayDouble array;

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
        array.resize(length);

    }

    public void copyTo(Tensor tensor)
    {
        tensor.resize(dims);
        ArrayDouble tA = tensor.getArray();
        System.arraycopy(array.v, 0, tA.v, 0, array.size());

    }

    public int[] dims;
    public ArrayDouble getArray()
    {
        return array;
    }

    public int size()
    {
        return array.size();
    }
    public void constant(double x)
    {
        array.constant(x);
    }


    public double get(int index)
    {
        return array.get(index);
    }

    // Same as get but NO bounds check
    public double at(int index)
    {
        return array.v[index];
    }

    public void set(double[] x)
    {
        array.set(x);
    }

    public void set(int index, double element)
    {
        array.set(index, element);
    }

    public void scale(double scalar)
    {
        array.scale(scalar);
    }
    public void add(double scalar)
    {
        array.add(scalar);
    }

    public double addi(int index, double scalar)
    {
        return array.addi(index, scalar);
    }
    public double multi(int index, double scalar)
    {
        return array.multi(index, scalar);
    }
    public void addn(double[] x)
    {
       array.addn(x);
    }
    public void multn(double[] x)
    {
        array.multn(x);
    }


    public Tensor transposeWeight4D()
    {

        int[] newDims = new int[dims.length];

        newDims[0] = dims[1];
        newDims[1] = dims[0];
        newDims[2] = dims[2];
        newDims[3] = dims[3];
        // If either feature map is one, no need to copy, memory is the same
        if (dims[0] == 1 || dims[1] == 1)
        {
            return new Tensor(array, newDims);
        }
        Tensor weightCopy = new Tensor(newDims);
        int sz = newDims[2] * newDims[3];
        for (int i = 0; i < newDims[0]; ++i)
        {
            for (int j = 0; j < newDims[1]; ++j)
            {
                for (int k = 0; k < sz; ++k)
                {
                    weightCopy.array.v[(i * newDims[1] + j) * sz + k] = this.array.v[(j * dims[1] + i) * sz + k];
                }
            }
        }
        return weightCopy;

    }

    public Tensor embed(int h, int w)
    {
        return embed(0, h, w);
    }
    public Tensor embed(int l, int h, int w)
    {
        final int tL = dims[0];
        final int tH = dims[1];
        final int tW = dims[2];
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
                    zpCube.array.v[((k + lStart) * oH + i + hStart) * oW + j + wStart] = array.v[(k * tH + i) * tW + j];
                }
            }
        }
        return zpCube;
    }
}


