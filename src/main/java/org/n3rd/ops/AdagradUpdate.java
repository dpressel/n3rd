package org.n3rd.ops;

import org.n3rd.Tensor;
import org.n3rd.layers.Layer;

/**
 * Created by dpressel on 4/7/16.
 */
public class AdagradUpdate implements Update
{
    protected double alpha;

    private static final double EPS = 1e-8;

    public AdagradUpdate(double alpha)
    {
        this.alpha = alpha;
    }
    @Override
    public void run(Layer layer, double eta, double lambda)
    {
        Tensor gg = layer.getWeightAccum();
        Tensor weights = layer.getParams();
        int wSz = weights.size();

        Tensor weightGrads = layer.getParamGrads();

        for (int i = 0; i < wSz; ++i)
        {
            if (weightGrads.get(i) == 0.0)
            {
                continue;
            }


            double gwi = weightGrads.get(i);
            double ggi = gg.get(i);
            gg.set(i, alpha * ggi + gwi*gwi);
            ggi = gg.get(i);

            double etaThis = eta / Math.sqrt(ggi + EPS);
            double delta = -etaThis * gwi;

            //double delta = -eta * gwi;
            double wi = weights.get(i) * (1 - eta * lambda);
            wi += delta;
            weights.set(i, wi);
            weightGrads.set(i, 0);

        }
    }

}
