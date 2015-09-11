n3rd
====

## Simple, no frills, easy to use (and understand) neural nets / deep learning in Java

I wanted a simple, easy to use, flexible (enough) neural net architecture for experiments, and at the same time, I wanted to see how far we can push Leon Bottou's ubiquitous SGD approach as a backbone for deep neural nets.  What I ended up is very simple, and does reuse a lot of the basic architecture (in the form of [sgdtk](https://github.com/dpressel/sgdtk/blob/master/README.md), my modular implementation of SGD for structured and unstructured predictions based on Bottou's sample code), keeping the actual code very minimal. The code also draws from various other sources of inspiration for its own contributions, particularly [Torch](https://github.com/torch).  Due to how the SGD framework is structured (primarily for linear classification problems), we are left handling backprop in the model, outside of the actual Learner, a necessary by-product (I think) of preserving the original structure.

For the time being, this code is implemented in pure CPU Java, though I believe other backends should be possible as well (I hope to address this in the near future).  Adagrad is currently used to train the network layers.

## Some samples

Some full sample programs will be added in the near future, but for now, a few snippets.  Due to how this is built, it might be useful to dive into [sgdtk](https://github.com/dpressel/sgdtk/blob/master/README.md) and understand how to train basic linear classifiers.

Here is how you can create a LeNet-5 like Neural Net

```java
static Learner createTrainer(Params params)
{
    NeuralNetModelFactory factory = new NeuralNetModelFactory();

    factory.addLayer(new SpatialConvolutionalLayer(6, 5, 5, new int[]{1,32,32}));
    factory.addLayer(new MaxPoolingLayer(2, 2, 6, 28, 28));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new SpatialConvolutionalLayer(16, 5, 5, new int[]{6,14,14}));
    factory.addLayer(new MaxPoolingLayer(2, 2, 16, 10, 10));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new SpatialConvolutionalLayer(128, 5, 5, new int[]{16,5,5}));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new FullyConnectedLayer(84, 128));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new FullyConnectedLayer(10, 84));
    factory.addLayer(new LogSoftMaxLayer());
    Learner learner = new SGDLearner(new ClassNLLLoss(), params.lambda, params.eta0, factory, new FixedLearningRateSchedule());
        return learner;
}

```

Just slurping in the MNIST files, you can do something like this:

```java

// Pad to 32x32
DatasetReader reader = new MNISTReader(ZERO_PAD);

long l0 = System.currentTimeMillis();
List<FeatureVector> trainingSet = reader.load(labelFile, imagesFile);

double elapsed = (System.currentTimeMillis() - l0) / 1000.;
System.out.println("Training data (" + trainingSet.size() + " examples) + loaded in " + elapsed + "s");

List<FeatureVector> evalSet = null;
if (params.evaly != null)
{
    labelFile = new File(params.evaly);
    imagesFile = new File(params.evalx);
    evalSet = reader.load(labelFile, imagesFile);
    Collections.shuffle(evalSet);
}

Learner learner = createTrainer(params);
Model model = learner.create(null);
double totalTrainingElapsed = 0.;

for (int i = 0; i < params.epochs; ++i)
{
    System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    System.out.println("EPOCH: " + (i + 1));

    Collections.shuffle(trainingSet);

    Metrics metrics = new Metrics();
    double t0 = System.currentTimeMillis();

    learner.trainEpoch(model, trainingSet);
    double elapsedThisEpoch = (System.currentTimeMillis() - t0) / 1000.;
    System.out.println("Epoch training time " + elapsedThisEpoch + "s");
    totalTrainingElapsed += elapsedThisEpoch;

    learner.eval(model, trainingSet, metrics);
    ExecUtils.showMetrics(metrics, "Training Set Eval Metrics");
    metrics.clear();

    if (evalSet != null)
    {
        learner.eval(model, evalSet, metrics);
        ExecUtils.showMetrics(metrics, "Test Set Eval Metrics");
        metrics.clear();
    }

}

System.out.println("Total training time " + totalTrainingElapsed + "s");
if (params.model != null)
{
    model.save(new FileOutputStream(params.model));
}

```

Here is a simple example of a Convolutional Net for binary sentence classification, e.g., for positive/negative sentiment analysis.  It assumes that the input are zero-padded sentences (making a wide convolution) of word vectors (300 here), and employs K-Average Folding to collapse the embeddings and K-Max pooling.

```java
static Learner createTrainer(Params params)
{
    Learner learner = new SGDLearner(new LogLoss(), params.lambda, params.eta0, new NeuralNetModelFactory(new Layer[] {
            // Emit 8 feature maps use a kernel width of 7 -- embeddings are 300 deep (L1)
            new TemporalConvolutionalLayer(4, 1, 7, 300),
            // Cut the embedding dim down to 75 by averaging adjacent embedding rows
            new AverageFoldingLayer(4, 300, 4),
            // Do K dynamic pooling grabbing the 3 highest values from each signal
            new KMaxPoolingLayer(3, 4, 75), new TanhLayer(),
            // 3 * 4 * 75 = 900
            new FullyConnectedLayer(100, 900), new TanhLayer(),
            new FullyConnectedLayer(1, 100), new TanhLayer() }));
    return learner;
}
```

Now, if we have some word2vec models laying around, let's get cracking... With some basic abstraction in how we read the data, it can be trained pretty much the same way....

```java

// Zero-padding for wide convolution
OrderedEmbeddedDatasetReader reader = new OrderedEmbeddedDatasetReader("D:/data/xdata/GoogleNews-vectors-negative300.bin", (7 - 1) / 2);

long l0 = System.currentTimeMillis();
List<FeatureVector> trainingSet = reader.load(trainFile);

ExecUtils.stats(trainingSet);

List<FeatureVector> evalSet = null;
if (params.eval != null)
{
    File evalFile = new File(params.eval);
    evalSet = reader.load(evalFile);
    Collections.shuffle(evalSet);
    ExecUtils.stats(evalSet);
}

Learner learner = createTrainer(params);

Model model = learner.create(null);
double totalTrainingElapsed = 0.;

for (int i = 0; i < params.epochs; ++i)
{
    System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    System.out.println("EPOCH: " + (i + 1));

    Collections.shuffle(trainingSet);

    Metrics metrics = new Metrics();
    double t0 = System.currentTimeMillis();

    learner.trainEpoch(model, trainingSet);//trainingSet.subList(0, 1000));
    double elapsedThisEpoch = (System.currentTimeMillis() - t0) / 1000.;
    System.out.println("Epoch training time " + elapsedThisEpoch + "s");
    totalTrainingElapsed += elapsedThisEpoch;

    learner.eval(model, trainingSet.subList(10000, 12000), metrics);
    ExecUtils.showMetrics(metrics, "Training Set Eval Metrics");
    metrics.clear();

    if (evalSet != null)
    {
        learner.eval(model, evalSet, metrics);
        ExecUtils.showMetrics(metrics, "Test Set Eval Metrics");
        metrics.clear();
    }

}
```
The underlying sgdtk library has good support for overlapped IO and processing, influenced by [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit), which makes it easy to process large amounts of data that would not fit into memory.  This is the recommended usage, unlike the above toy examples.  Note also that, unlike the sgdtk base, where we use a OvA MultiClassWeightModel for multi-class decisions, in n3rd, we are using a LogSoftMax with ClassNLLLoss instead to accomplish this without requiring that extra overhead.
