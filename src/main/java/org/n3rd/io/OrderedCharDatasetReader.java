package org.n3rd.io;

import org.sgdtk.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.stream.IntStream;

public class OrderedCharDatasetReader implements DatasetReader
{
    int largestVectorSeen = 0;

    private int paddingSzPerSide;
    private int lineNumber = 0;

    BufferedReader reader;
    int HASH_SIZE = 8192;
    int MAX_CHARS = 140;
    int NBITS = 13;
    HashFeatureEncoder hashFeatureEncoder = new HashFeatureEncoder(NBITS);
    FeatureNameEncoder labelEncoder;

    @Override
    public int getLargestVectorSeen()
    {
        return largestVectorSeen;
    }

    // Only use this if your data is numerically labeled (which should start at 1)
    public OrderedCharDatasetReader(int paddingSzPerSide) throws IOException
    {
        this(paddingSzPerSide, null);
    }
    // Label encoder MUST be populated
    public OrderedCharDatasetReader(int paddingSzPerSide, FeatureNameEncoder labelEncoder) throws IOException
    {
        this.labelEncoder = labelEncoder == null ? new LazyFeatureDictionaryEncoder(): labelEncoder;
        this.paddingSzPerSide = paddingSzPerSide;

    }

    /**
     * Open a file for reading.  All files are read only up to maxFeatures.
     * @param file An SVM light type file
     * @throws IOException
     */
    @Override
    public final void open(File... file) throws IOException
    {
        lineNumber = 0;
        reader = new BufferedReader(new FileReader(file[0]));
    }

    /**
     * Close the currently loaded file
     * @throws IOException
     */
    @Override
    public final void close() throws IOException
    {
        reader.close();
    }

    /**
     * Slurp the entire file into memory.  This is not the recommended way to read large datasets, use
     * {@link #next()} to stream features from the file one by one.
     *
     * @param file An SVM light type file
     * @return One feature vector per line in the file
     * @throws IOException
     */
    @Override
    public final List<FeatureVector> load(File... file) throws IOException
    {

        open(file);

        List<FeatureVector> fvs = new ArrayList<FeatureVector>();
        //CRSFactory factory = new CRSFactory();

        FeatureVector fv;

        while ((fv = next()) != null)
        {
            fvs.add(fv);
        }

        System.out.println("Read " + lineNumber + " lines");
        close();
        // Read a line in, and then hash it into the bin vector
        return fvs;
    }

    /**
     * Get the next feature vector in the file
     *
     * @return The next feature vector, or null, if we are out of lines
     * @throws IOException
     */
    public final FeatureVector next() throws IOException
    {
        final String line = reader.readLine();

        if (line == null)
        {
            return null;
        }


        lineNumber++;

        //System.out.println(lineNumber);
        final StringTokenizer tokenizer = new StringTokenizer(line, " \t");
        String strLabel = tokenizer.nextToken();
        Integer label;
        try
        {
            label = Integer.valueOf(strLabel);
        }
        catch (NumberFormatException numEx)
        {
            label = labelEncoder.indexOf(strLabel);
            if (label == null)
            {
                return next();
            }
            // This is due to the zero offset assigned by the lazy encoder, we want 1-based
            label++;
        }
        List<Integer> lookup = new ArrayList<>();

        int charsUsed = 0;
        while (tokenizer.hasMoreTokens() && charsUsed < MAX_CHARS)
        {
            String word = tokenizer.nextToken();
            int wSz = word.length();
            for (int i = 0; i < wSz - 1; ++i)
            {
                int fIdx = hashFeatureEncoder.lookupOrCreate(word.substring(i, i + 1));

                lookup.add(fIdx);
                ++charsUsed;
                if (charsUsed == MAX_CHARS)
                {
                    break;
                }
            }


        }
        int sentenceSz = lookup.size(); //charsUsed

        if (sentenceSz < 1)
        {
            return next();
        }

        int pitch = 2 * paddingSzPerSide + sentenceSz;

        // Vector is flat, words x esz
        DenseVectorN x = new DenseVectorN((2*paddingSzPerSide + sentenceSz) * HASH_SIZE);

        final FeatureVector fv = new FeatureVector(label, x);

        for (int i = 0, ibase = 0; i < sentenceSz; ++i, ibase += HASH_SIZE)
        {
            int j = lookup.get(i);
            x.set(j * pitch + i + paddingSzPerSide, 1.0);
        }

        fv.getX().organize();
        return fv;
    }

}
