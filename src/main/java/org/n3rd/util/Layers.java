package org.n3rd.util;

import org.n3rd.layers.Layer;

import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

public class Layers
{
    final static Set<String> FILTER_METHODS;
    final static String[] FILTER_METHODS_LIST = {
            "getParamGrads",
            "getBiasGrads",
            "getClass",
    };
    static {
        FILTER_METHODS = new HashSet<String>(Arrays.asList(FILTER_METHODS_LIST));
    }

    public static Layer toLayer(Map<String, Object> params) throws IOException
    {
        try
        {
            // Invoke the default constructor for this class
            String className = (String) params.get("Name");
            Class javaClass = Class.forName(className);
            Constructor constructor = javaClass.getConstructor();
            Layer layer = (Layer)constructor.newInstance();

            for (String field : params.keySet())
            {
                if ("Name".equals(field))
                {
                    continue;
                }

                Object object = params.get(field);
                if (object == null)
                {
                    continue;
                }

                String setMethod = "set" + field;
                Method method = javaClass.getDeclaredMethod(setMethod, object.getClass());
                method.invoke(layer, object);
            }


            return layer;
        }
        catch (NoSuchMethodException classEx)
        {
            throw new IOException(classEx);
        }
        catch (ClassNotFoundException classEx)
        {
            throw new IOException(classEx);
        }
        //catch (IntrospectionException introEx)
        //{
        //    throw new IOException(introEx);
        //}
        catch (IllegalAccessException illAxEx)
        {
            throw new IOException(illAxEx);
        }
        catch (IllegalArgumentException illArgEx)
        {
            throw new IOException(illArgEx);
        }
        catch (InvocationTargetException invokeEx)
        {
            throw new IOException(invokeEx);
        }
        catch (InstantiationException instEx)
        {
            throw new IOException(instEx);
        }
    }

    public static Map<String, Object> toParams(Layer layer) throws IOException
    {
        try
        {
            Map<String, Object> params = new HashMap<String, Object>();
            for (PropertyDescriptor propertyDescriptor :
                    Introspector.getBeanInfo(layer.getClass()).getPropertyDescriptors())
            {
                Method readMethod = propertyDescriptor.getReadMethod();
                if (FILTER_METHODS.contains(readMethod.getName()))
                {
                    continue;
                }
                String name = readMethod.getName().replace("get", "");
                Object output = readMethod.invoke(layer);
                params.put(name, output);
            }
            params.put("Name", layer.getClass().getName());
            return params;
        }
        catch (IntrospectionException introEx)
        {
            throw new IOException(introEx);
        }
        catch (IllegalAccessException illAxEx)
        {
            throw new IOException(illAxEx);
        }
        catch (InvocationTargetException invokeEx)
        {
            throw new IOException(invokeEx);
        }
    }

}
