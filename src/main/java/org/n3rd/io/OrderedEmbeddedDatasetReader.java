package org.n3rd.io;

import org.sgdtk.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Produce a temporal feature vector
 *
 * A temporal vector is going to have the word vectors side by side.
 * ["this", "is", "a", "car"] => [wv_this, wv_is, wv_a, wv_car];
 *
 * Once this has happened, we have a temporal, feature free array
 * This can assume a distributed representation, which would make it easier.  Otherwise we would need a one-hot
 * vector at every position.
 * @author dpressel
 */
public class OrderedEmbeddedDatasetReader implements DatasetReader
{
    int largestVectorSeen = 0;
    Word2VecModel word2vecModel;

    FeatureNameEncoder labelEncoder;
    private int embeddingSize;
    private int paddingSzPerSide;
    private int lineNumber = 0;

    int MAX_FEATURES = 4096;
    @Override
    public int getLargestVectorSeen()
    {
        return largestVectorSeen;
    }

    public OrderedEmbeddedDatasetReader(String embeddings) throws IOException
    {
        this(embeddings, 0, null);
    }
    public OrderedEmbeddedDatasetReader(String embeddings, int paddingSzPerSide, FeatureNameEncoder labelEncoder) throws IOException
    {
        word2vecModel = Word2VecModel.loadWord2VecModel(embeddings);
        this.paddingSzPerSide = paddingSzPerSide;
        this.labelEncoder = labelEncoder == null ? new LazyFeatureDictionaryEncoder(): labelEncoder;

    }

    BufferedReader reader;

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

        long lv = word2vecModel.getSize();
        embeddingSize = (int)lv;
        lineNumber++;

        //System.out.println(lineNumber);
        final StringTokenizer tokenizer = new StringTokenizer(line, " \t");
        final int lastIdxTotal = MAX_FEATURES - 1;

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

        int idx = 0;
        List<ArrayFloat> lookup = new ArrayList<ArrayFloat>();
        while (tokenizer.hasMoreTokens())
        {
            ++idx;
            String word = tokenizer.nextToken().toLowerCase();
            try
            {
                float[] vec = word2vecModel.getVec(word);
                ArrayFloat af = new ArrayFloat(vec);
                lookup.add(af);
            }
            catch (Exception ex)
            {
                continue;
            }

            if (lastIdxTotal > 0 && idx > lastIdxTotal)
                continue;

        }
        int sentenceSz = lookup.size();

        if (sentenceSz < 1)
        {
            return next();
        }

        if (idx >= largestVectorSeen)
        {
            largestVectorSeen = idx;
        }

        int pitch = 2 * paddingSzPerSide + sentenceSz;
        // Vector is flat, words x esz
        DenseVectorN x = new DenseVectorN((2*paddingSzPerSide + sentenceSz) * embeddingSize);

        final FeatureVector fv = new FeatureVector(label, x);
        //int padOffset = paddingSzPerSide * embeddingSize;


        for (int j = 0; j < embeddingSize; ++j)
        {
            for (int i = 0, ibase = 0; i < sentenceSz; ++i, ibase += embeddingSize)
            {
                double wordVecj = lookup.get(i).get(j);
                x.set(j * pitch + i + paddingSzPerSide, wordVecj);
            }
        }


        fv.getX().organize();
        return fv;
    }

    public int getEmbeddingSize()
    {
        return embeddingSize;
    }

    public void setEmbeddingSize(int embeddingSize)
    {
        this.embeddingSize = embeddingSize;
    }
}
