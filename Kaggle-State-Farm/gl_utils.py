import graphlab as gl
from config import GL_MODEL

def get_sf(func, times=1):
    df, flatten = func()
    export = gl.SFrame(df)
    if times > 1:
        length = flatten.shape[0]
        steps = range(0, len(flatten), len(flatten) / times)
        ret = gl.SFrame()
        for i in range(len(steps) - 1):
            ret = ret.append(gl.SFrame(flatten[steps[i]:steps[i + 1]]))
        ret = ret.append(gl.SFrame(flatten[steps[-1]:]))
        del flatten
    else:
        ret = gl.SFrame(flatten)
    export['conv'] = ret['X1']
    return export

def load_model():
    boosting = gl.load_model(GL_MODEL)
    return boosting

def result_sf2df(gl_model, sf):
    pred = gl_model.predict_topk(sf, k=10)
    class_proba = pred.to_dataframe().pivot_table('probability',index='id',columns='class')
    df = sf.to_dataframe().join(class_proba).drop('conv',axis=1)
    return df
