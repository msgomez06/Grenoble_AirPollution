  *	)\?ZA2T
Iterator::Prefetch::Generator?̯? ?_@!O???#rI@)?̯? ?_@1O???#rI@:Preprocessing2?
aIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::Prefetch::FlatMap[0]::Generator???x^@!?:?!;?H@)???x^@1?:?!;?H@:Preprocessing2I
Iterator::Prefetch???.5B??!~?(?????)???.5B??1~?(?????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???Y??!ѹ?ku^??)???Y??1ѹ?ku^??:Preprocessing2n
7Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard?9\?=???!{[u<??)?E??(&??1g??g???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism2???A???!|???@???)Q?????1W??????:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch?-?R???!?т???)X??????1??B?!,~?:Preprocessing2F
Iterator::Model]~p>u??!?g
?bt??)[y??????1-dHA?x?:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::Prefetch::FlatMap32?]?y^@!??d?݂H@).??H??1?S??5Vt?:Preprocessing2?
JIterator::Model::MaxIntraOpParallelism::Prefetch::Shard::Rebatch::Prefetch??V|C???!-??j??o?)??V|C???1-??j??o?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.