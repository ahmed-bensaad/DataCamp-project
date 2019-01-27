from data_gens import load_annotations, load_data
import tensorflow as tf
import tensornets as nets


def train():
    dirr = './DataCamp-project/Data/'

    anns = load_annotations(dirr)
    trains = load_data(anns)
    inputs = tf.placeholder(tf.float32, [None, 128, 128, 3])
    is_training = tf.placeholder(tf.bool)
    model = nets.YOLOv2(inputs, nets.Darknet19, is_training=is_training, classes = 43)
    print("model loaded")
    step = tf.Variable(0, trainable=False)
    lr = tf.train.piecewise_constant(
        step, [50, 1960, 5000, 10000,30000,],
        [2.5e-4, 2.5e-4, 1e-4, 1e-4,  1e-5, 1e-5])
    train = tf.train.AdamOptimizer(lr).minimize(model.loss,
                                                    global_step=step)
    print("Calculations initialized")



    with tf.Session() as sess:

    # Load Darknet19
        sess.run(tf.global_variables_initializer())
        sess.run(model.stem.pretrained())
        print("Darknet loaded")
    # Note that there are 16551 images (5011 in VOC07 + 11540 in VOC12).
    # When the mini-batch size is 48, 1 epoch consists of 344(=16551/48) steps.
    # Thus, 233 epochs will cover 80152 steps.
        losses = []
        print("Begin training")
        for i in range(10):

            _t = time.time()
            for (imgs, metas) in trains:
                # `trains` returns None when it covers the full batch once
                if imgs is None:
                    break
                metas.insert(0, model.preprocess(imgs))  # for `inputs`
                metas.append(True)  # for `is_training`
                outs = sess.run([train, model.loss],
                                dict(zip(model.inputs, metas)))
                losses.append(outs[1])

        # Report step, learning rate, loss, weight decay, runtime
                print('\r','***** %d %.5f %.5f %.5f %.5f *****' %
                    (sess.run(step), sess.run(lr),
                    losses[-1], sess.run(tf.losses.get_regularization_loss()),
                    time.time() - _t),end="")
            #save model every 500 step
            print('epoch = {}. Saving weights'.format(i))
            np.save('weights_v2', sess.run(model.get_weights())) 


if __name__=='__main__':
    train()