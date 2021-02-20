Tuning Transformer Pipelines
----------------------------

Many approaches will have parameters that require tuning. PyTerrier helps to achieve this by proving a grid search
which can tune one or more parameters using a particular evaluation metric. `pt.GridSearch()` is the method of achiving this.

Pre-requisites
==============

GridSearch makes several assumption:
 - the transformer stages in your pipeline are uniquely identified, by setting of an `id=` keyword argument during their construction.
 - the parameters that you wish to tune are available as instannce variables within the transformers, or that the transformer responds suitably to `set_parameter()`. 
 - changing the relevant parameters has an impact upon subsequent calls to `transform()`.

Note that transformers implemented using pt.apply functions cannot address the second requirement, as any parameters are captured 
naturally within the closure, and not as instances variable of the transformer.

API
===

.. autofunction:: pyterrier.GridSearch()

Examples
========

Tuning BM25
~~~~~~~~~~~

When using BatchRetrieve, the b parameter of BM25 can be controled using the "c" control. 
We must give this control an initial value when contructing the BatchRetrieve instance::

    BM25 = pt.BatchRetrieve(index, wmodel="BM25", id="bm25",  controls={"c" : 0.75})
    pt.GridSearch(BM25, vaswani.get_topics().head(10), vaswani.get_qrels(), {"bm25" : {"c" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]}})


Tuning BM25 and RM3
~~~~~~~~~~~~~~~~~~~

The query expansion transformer in pt.rewrite have parameters controlling the number of feedback documents and expansion terms, namely:
 - fb_terms -- the number of terms to add to the query.
 - fb_docs -- the size of the pseudo-relevant set. 


A full tuning of BM25 and RM3 can be achieved as thus::

    bm25_for_qe = pt.BatchRetrieve(index, wmodel="BM25", controls={"c" : 0.75}, id="bm25_qe")
    pipe_qe = bm25_for_qe >> pt.rewrite.RM3(index, fb_terms=10, fb_docs=3, id="rm3") >> bm25_for_qe

    param_map = {
            "bm25_qe" : { "c" : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 ]},
            "bo1" : { 
                "fb_terms" : list(range(1, 12, 3)), 
                "fb_docs" : list(range(2, 30, 6))
            }
    }
    _, best_param_map = pt.GridSearch(pipe_qe, topics, qrels, param_map)
