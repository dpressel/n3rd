package org.n3rd.util;

public class IntCube
{
    public IntCube(int l, int h, int w)
    {
        this.l = l;
        this.h = h;
        this.w = w;
        d = new int[l * w * h];
    }


    public int l;
    public int h;
    public int w;
    public int[] d;

    public void reset(int x)
    {
        for (int i = 0; i < d.length; ++i)
        {
            d[i] = x;
        }
    }
}


