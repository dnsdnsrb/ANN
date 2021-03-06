import tensorflow as tf

# import RW.svhn_mlp
# import RW.mnist_mlp
import RW.cifar_mlp
# import RW.news_mlp

# import RW.news_rw
# import RW.mnist_rw
# import RW.cifar_rw
# import RW.svhn_rw


#dropout 설정이 잘못된채로 데이터를 새로 뽑아넴. 0 혹은 1이 아닌 dropout을 모두 재시험할 것.
# layers.dropout은 기본적으로 training=False가 되어있어 적용이 안됨. palceholder를 사용한다면 nn으로 바꾸는게 더 편할듯

# net = RW.cifar_mlp.Network()
# net.run(100000)
#
# tf.reset_default_graph()
#
# net = RW.cifar_mlp.Network(regularization=True)
# net.run(100000)
#
# tf.reset_default_graph()
#
# net = RW.cifar_mlp.Network(batch_normalization=True)
# net.run(100000)
#
# tf.reset_default_graph()

net = RW.cifar_mlp.Network(droprate=0.5)
net.run(100000)