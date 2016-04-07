package org.n3rd.ops;

import org.n3rd.Tensor;
import org.n3rd.layers.Layer;

/**
 * Created by dpressel on 4/7/16.
 */
public class SGDWithMomentumUpdate implements Update
{
    private double alpha;

    public SGDWithMomentumUpdate(double alpha)
    {
        this.alpha = alpha;
    }

    @Override
    public void run(Layer layer, double eta, double lambda)
    {
        Tensor last = layer.getWeightAccum();
        Tensor weights = layer.getParams();
        int wSz = weights.size();

        Tensor weightGrads = layer.getParamGrads();

        for (int i = 0; i < wSz; ++i)
        {
            double gwi = weightGrads.get(i);
            double lasti = last.get(i);

            double delta = -eta * gwi + alpha * lasti;
            last.set(i, delta);

            // weight decay
            double wi = weights.get(i) * (1 - eta * lambda);
            wi += delta;
            weights.set(i, wi);
            weightGrads.set(i, 0);
        }


    }
}
