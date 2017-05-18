import tensorflow as tf
import deep_n_wide as wndt

def create_feature_columns():
    # Sparse base columns.
    brands = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="brands", hash_bucket_size=10000)
    province = tf.contrib.layers.sparse_column_with_hash_bucket(
      "province", hash_bucket_size=100)
    
    # Wide columns and deep columns.
    wide_columns = [brands, province]
    
    for i in range(0, wndt.APP_NUM):
      appName = 'app' + str(i)
      wide_columns.append(tf.contrib.layers.real_valued_column(appName))

    # Continuous base columns.
    #deep_columns = [       
    #  tf.contrib.layers.embedding_column(brands, dimension=8),
    #  tf.contrib.layers.embedding_column(province, dimension=8)
    #]
    
    #for i in range(0, wndt.APP_NUM):
    #  appName = 'app' + str(i)
    #  deep_columns.append(tf.contrib.layers.real_valued_column(appName))
      
    label = tf.contrib.layers.real_valued_column("gender", dtype=tf.int64)
    wide_columns.append(label)
    return wide_columns

def input_fn(mode, data_file, batch_size):
    try:
        input_features = create_feature_columns()
        features = tf.contrib.layers.create_feature_spec_for_parsing(input_features)

        feature_map = tf.contrib.learn.io.read_batch_record_features(
            file_pattern=[data_file],
            batch_size=batch_size,
            features=features,
            name="read_batch_features_{}".format(mode))
        #sess = tf.InteractiveSession()
        #print(feature_map["age"].eval())
        #sess.close()
      #  with tf.Session() as sess:
        #    print(sess.run(feature_map["age"]))
       #     print(feature_map["age"].eval())

        # a = feature_map
        # sess = tf.Session()
        # sess.run(a)
        x = tf.Print(feature_map['age'], [feature_map['age']])
        print('128')




        target = feature_map.pop("brands")
    except Exception as e:
        raise e

    return feature_map, target