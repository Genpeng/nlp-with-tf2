# coding=utf-8

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def check(filepath, batch_size):
    with tf.Session() as sess:
        filenames = [filepath]
        ds = tf.data.TFRecordDataset(filenames, compression_type="GZIP")
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=batch_size * 10)
        iterator = tf.data.make_one_shot_iterator(ds)
        batch_data = iterator.get_next()
        res = sess.run(batch_data)  # ndarray

        serialized_example = res[0]  # bytes
        example = tf.train.Example.FromString(serialized_example)
        features = example.features

        print(f"the features of file '{filepath}' are as follows:")
        for key in features.feature:
            print("=" * 100)

            feature = features.feature[key]
            if len(feature.bytes_list.value) > 0:
                feat_type = "bytes_list"
                feat_value = feature.bytes_list.value
            if len(feature.float_list.value) > 0:
                feat_type = "float_list"
                feat_value = feature.float_list.value
            if len(feature.int64_list.value) > 0:
                feat_type = "int64_list"
                feat_value = feature.int64_list.value

            out_str = f"{key}: {feat_type}, {feat_value}"
            print(out_str)


if __name__ == "__main__":
    check("/Users/gerry.xu/Downloads/part-r-01014.gz", batch_size=8)
