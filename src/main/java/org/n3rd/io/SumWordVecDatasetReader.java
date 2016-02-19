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
 * Produce a sum of BoW features
 * <p>
 * This is going to add the component wise features which will basically be a pre-projection layer and will cause a small
 * number of inputs.
 *
 * @author dpressel
 */
public class SumWordVecDatasetReader implements DatasetReader
{

    Word2VecModel word2vecModel;
    private long embeddingSize;
    private int lineNumber = 0;


    HashFeatureEncoder hashFeatureEncoder = new HashFeatureEncoder(24);

    FeatureNameEncoder labelEncoder;

    @Override
    public int getLargestVectorSeen()
    {
        return (int) embeddingSize;
    }


    public SumWordVecDatasetReader(String embeddings) throws IOException
    {
        this(embeddings, null);
    }
    public SumWordVecDatasetReader(String embeddings, FeatureNameEncoder labelEncoder) throws IOException
    {
        word2vecModel = Word2VecModel.loadWord2VecModel(embeddings);
        this.embeddingSize = word2vecModel.getSize();
        this.labelEncoder = labelEncoder == null ? new LazyFeatureDictionaryEncoder(): labelEncoder;
    }

    BufferedReader reader;

    /**
     * Open a file for reading.  All files are read only up to maxFeatures.
     *
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
     *
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
    public final List<FeatureVector> load(File... file) throws IOException
    {

        open(file);

        List<FeatureVector> fvs = new ArrayList<FeatureVector>();

        FeatureVector fv;

        while ((fv = next()) != null)
        {
            fvs.add(fv);
        }

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

        DenseVectorN x = new DenseVectorN((int)embeddingSize);
        ArrayDouble xArray = x.getX();
        while (tokenizer.hasMoreTokens())
        {
            String word = tokenizer.nextToken().toLowerCase();

            float[] af = word2vecModel.getVec(word);
            for (int j = 0; j < xArray.size(); ++j)
            {
                double vecj = af[j];
                xArray.addi(j, vecj);
            }
        }

        // Vector is flat, words x esz
        final FeatureVector fv = new FeatureVector(label, x);
        fv.getX().organize();
        return fv;
    }

    public int getEmbeddingSize()
    {
        return (int)embeddingSize;
    }

    public void setEmbeddingSize(int embeddingSize)
    {
        this.embeddingSize = embeddingSize;
    }
}
