package org.n3rd.io;

import org.n3rd.Tensor;
import org.sgdtk.DenseVectorN;
import org.sgdtk.FeatureVector;
import org.sgdtk.io.DatasetReader;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

public class MNISTReader implements DatasetReader
{

    RandomAccessFile imageFile;
    RandomAccessFile labelFile;

    int numImages;
    int numRows;
    int numCols;
    int current;
    int zeroPadding;
    public MNISTReader(int zp)
    {
        zeroPadding = zp;
    }

    /**
     * Open a file for reading.  All files are read only up to maxFeatures.
     * @param file label file first! Then second arg is image file
     * @throws IOException
     */
    public final void open(File... file) throws IOException
    {

        current = 0;

        labelFile = new RandomAccessFile(file[0], "r");
        int x = labelFile.readInt();
        if (x != 2049)
        {
            throw new IOException("Bad magic");
        }
        int numLabels = labelFile.readInt();


        imageFile = new RandomAccessFile(file[1], "r");
        x = imageFile.readInt();
        if (x != 2051)
        {
            throw new IOException("Bad magic");
        }
        numImages = imageFile.readInt();
        if (numLabels != numImages)
        {
            throw new IOException("Label/image mismatch!");
        }
        numRows = imageFile.readInt();
        numCols = imageFile.readInt();

    }

    /**
     * Close the currently loaded file
     * @throws IOException
     */
    public final void close() throws IOException
    {
        current = numCols = numRows = 0;
        imageFile.close();
        labelFile.close();
    }
    @Override
    public List<FeatureVector> load(File... file) throws IOException
    {
        open(file);

        List<FeatureVector> fvs = new ArrayList<FeatureVector>();
        //CRSFactory factory = new CRSFactory();

        FeatureVector fv;

        while ((fv = next()) != null)
        {
            fvs.add(fv);
        }

        System.out.println("Read " + numImages + " images");
        close();
        // Read a line in, and then hash it into the bin vector
        return fvs;
    }

    @Override
    public FeatureVector next() throws IOException
    {
        if (current == numImages)
        {
            return null;
        }

        Tensor t = readImage();
        DenseVectorN dv = new DenseVectorN(t.getArray().v);
        double label = readLabel();

        FeatureVector fv = new FeatureVector(label, dv);
        ++current;

        return fv;
    }

    private double readLabel() throws IOException
    {

        byte b = labelFile.readByte();
        int ati = ((int) b) & 0xFF;
        return (double)(ati + 1.0);
    }

    private Tensor readImage() throws IOException
    {
        final int numBytes = getLargestVectorSeen();

        byte[] buffer = new byte[numBytes];
        //int offset = numBytes * current;
        imageFile.read(buffer);

        Tensor tensor = new Tensor(1, numRows, numCols);

        for (int i = 0; i < numBytes; ++i)
        {
            int ati = ((int) buffer[i]) & 0xFF;
            tensor.set(i, ati / 255.0);
        }
        return tensor.embed(0, zeroPadding, zeroPadding);
    }

    @Override
    public int getLargestVectorSeen()
    {
        return numRows * numCols;
    }
}
