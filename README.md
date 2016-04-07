n3rd
====

## Simple, no frills, easy to use (and understand) neural nets / deep learning in Java

I spent a lot of time with Leon Bottou's SGD work and sample code, and found it useful for linear classifiers. For those that are familiar with that code, it may not be surprising that I also like [Torch](https://github.com/torch), which shares some similarities, but is a lot more general purpose and reusable.  I wanted a simple, easy to use, flexible (enough) neural net architecture for experiments, and at the same time, use my own re-implementation as a modular library as a backbone for deep neural nets.  I hoped that the result would demonstrate the leap from an SGD-based linear classifier to deep (and shallow) neural nets, and provide a clean SoC between the two.  I'm unclear if I achieved that but...

What I ended up is very simple, and does reuse a lot of the basic architecture, in the form of [sgdtk](https://github.com/dpressel/sgdtk/blob/master/README.md), keeping the actual code very minimal. The code also draws from various other sources of inspiration for its own contributions, particularly Torch.  Due to how the SGD framework is structured (primarily for linear classification problems), we are left handling backprop in the model, outside of the actual Learner, a necessary by-product (I think) of preserving the original structure.  Neural nets are trained using an Update object, with adagrad and SGD (with optional momentum) variants supported.

For the time being, this code is implemented in pure CPU Java, though I am working on a GPU backend.  However, I have reimplemented most of this code in nearly identical C++, with a CUDA backend under the name [n3rd-cpp](https://github.com/dpressel/n3rd-cpp).  The C++, in several cases, is much faster than the Java (CPU and GPU), though this varies depending on the operation, and both are still works in progress.  In general, for best performance in convolution, you'll want to use the blas-backed implementations in either case.

## Some samples

Some full sample programs will be added in the near future, but for now, a few snippets.  Due to how this is built, it might be useful to dive into [sgdtk](https://github.com/dpressel/sgdtk/blob/master/README.md) and understand how to train basic linear classifiers.

Here is how you can create a LeNet-5 like Neural Net with multiclass output that can achieve around 99% accuracy on the MNIST task:

```java
static Learner createTrainer(Params params)
{
    NeuralNetModelFactory factory = new NeuralNetModelFactory();
    // You can also just inline the layers as a Layer array
    // see examples below.
    factory.addLayer(new SpatialConvolutionalLayerBlas(6, 5, 5, new int[]{1,32,32}));
    factory.addLayer(new MaxPoolingLayer(2, 2, 6, 28, 28));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new SpatialConvolutionalLayerBlas(16, 5, 5, new int[]{6,14,14}));
    factory.addLayer(new MaxPoolingLayer(2, 2, 16, 10, 10));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new SpatialConvolutionalLayerBlas(128, 5, 5, new int[]{16,5,5}));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new FullyConnectedLayer(84, 128));
    factory.addLayer(new TanhLayer());
    factory.addLayer(new FullyConnectedLayer(10, 84));
    factory.addLayer(new LogSoftMaxLayer());

    // default is to use adagrad
    // factory.setUpdate(new AdagradUpdate(1.0);

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

A lot of tooling for deep learning doesnt provide a ton of flexibility for 1D convolutional nets.  That space is constantly evolving, but I wanted to support several different styles of CNNs that might be suitable for sentence processing, including Collobert/Weston-style nets and Kalchbrenner/Blunsom-style nets.  Additionally, n3rd currently supports several types of 1D convolution including standard form, as a single matrix multiply in unrolled form using BLAS (see [High Perf. CNN for Document Processing - Chellapilla, Puri, Simard](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=BB0ABD1378F88436F00A9ABE61F36DBC?doi=10.1.1.137.482&rep=rep1&type=pdf) ), and in an FFT form.   

Here is a simple example of a Kalchbrenner-style Convolutional Net for binary sentence classification, e.g., for positive/negative sentiment analysis.  It assumes that the input are zero-padded sentences (making a wide convolution) of word vectors (300 here), preserving embeddings through the convolution, and then employing K-Average Folding to collapse the embeddings and K-Max pooling.

```java
static Learner createTrainer(Params params)
{
    Learner learner = new SGDLearner(new LogLoss(), params.lambda, params.eta0, new NeuralNetModelFactory(new Layer[] {
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

Running a Collobert-style Convolutional Net is also simple -- here we simply dont supply the last argument, and the embeddings are treated as input feature maps (analagous to the LeNet5 model above).  Additionally, the C/W-type models use simple max-over time pooling

```java
static Learner createTrainer(Params params)
{
    NeuralNetModelFactory  nnmf = new NeuralNetModelFactory(new Layer[] {
        new TemporalConvolutionalLayerBlas(100, 300, FILTER_WIDTH),
        new MaxOverTimePoolingLayer(100), new DropoutLayer(0.5), 
        new FullyConnectedLayer(1, 100), new TanhLayer() });
        SGDLearner learner = new SGDLearner(new LogLoss(), params.lambda, params.eta0, nnmf);

    return learner;
}

```

Now, if we have some word2vec models laying around, let's get cracking... With some basic abstraction in how we read the data, it can be trained pretty much the same way....

```java

// Zero-padding for wide convolution
OrderedEmbeddedDatasetReader reader = new OrderedEmbeddedDatasetReader("D:/data/xdata/GoogleNews-vectors-negative300.bin", (FILTER_WIDTH - 1) / 2);

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

    model.setIsTraining(true); // for dropout
    learner.trainEpoch(model, trainingSet);
    double elapsedThisEpoch = (System.currentTimeMillis() - t0) / 1000.;
    System.out.println("Epoch training time " + elapsedThisEpoch + "s");
    totalTrainingElapsed += elapsedThisEpoch;

    model.setIsTraining(false); // for dropout
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
```
The underlying sgdtk library has good support for overlapped IO and processing which makes it easy to process large amounts of data that would not fit into memory.  Here is a snippet showing how to do online training using an internal ring buffer:

```

OverlappedTrainingRunner asyncTrainer = new OverlappedTrainingRunner(learner);
asyncTrainer.setEpochs(params.epochs);
asyncTrainer.setProbAdd(0.4);
asyncTrainer.setDense(true);

File evalFile = new File(params.eval);
            OrderedEmbeddedDatasetReader reader = new OrderedEmbeddedDatasetReader(params.embeddings, (FILTER_WIDTH - 1) / 2, null, 5, 140);
            
final List<FeatureVector> evalSet = params.eval != null ? reader.load(evalFile): null;

// Set a callback for epoch ends
asyncTrainer.addListener(new TrainingEventListener()
{
    double totalTrainingElapsed = 0;
    @Override
    public void onEpochEnd(Learner learner, Model model, double s)
    {
        Metrics metrics = new Metrics();
        totalTrainingElapsed += s;
        if (evalSet != null)
        {
            NeuralNetModel nnm = (NeuralNetModel)model;
            nnm.setIsTraining(false);
            learner.eval(model, evalSet, metrics);
            ExecUtils.showMetrics(metrics, "Test Set Eval Metrics");
            metrics.clear();
            System.out.println("Total training time " + totalTrainingElapsed + "s");
            nnm.setIsTraining(true);
        }
    }

});

asyncTrainer.start();

File trainFile = new File(params.train);
reader.open(trainFile);

FeatureVector fv;

int i = 0;
for (; (fv = reader.next()) != null; ++i)
{
    asyncTrainer.add(fv);
    if (i % 1000 == 0)
    {
        ExecUtils.rewriteInplace(System.out, "Loaded " + String.valueOf(i));
    }
}
       
ExecUtils.rewriteInplace(System.out, "Loaded " + String.valueOf(i));
System.out.println();
Model model = asyncTrainer.finish();            
reader.close();
```

Note also that, unlike the sgdtk base, where we use a OvA MultiClassWeightModel for multi-class decisions, in n3rd, we are using a LogSoftMax with ClassNLLLoss instead to accomplish this without requiring that extra overhead.

