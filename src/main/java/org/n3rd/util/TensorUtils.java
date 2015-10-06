package org.n3rd.util;

import org.n3rd.Tensor;

public class TensorUtils
{
    public static void printT2(Tensor t2)
    {
        for (int i = 0; i < t2.dims[0]; ++i)
        {
            for (int j = 0; j < t2.dims[1]; ++j)
            {
                System.out.print(String.format("%.2f\t", t2.d[j * t2.dims[0] + i]));
            }
            System.out.println();
        }
        System.out.println();
    }


    public static void printT3(Tensor tensor)
    {
        final int bands = tensor.dims[0];
        final int rows = tensor.dims[1];
        final int cols = tensor.dims[2];
        for (int k = 0; k < bands; ++k)
        {
            System.out.println("k = " + k);
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    System.out.print(String.format("%.2f\t", tensor.d[(k * rows + i) * cols + j]));
                }
                System.out.println();
            }
            System.out.println();
        }

    }
}
