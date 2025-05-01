#!/usr/bin/env python3

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import tensorflow as tf
import math


def xavier_init(n, m):
    dev = 1 / math.sqrt((n + m) / 2)
    return tf.Variable(tf.random.normal([n, m], dtype=float_type) * dev)


def net_u(t, x):
    t1 = t[:, :, None] * W1_t[None, :]
    t2 = t[:, :, None] * W2_t[None, :]
    x1 = x[:, :, None] * W1_x[None, :]
    H1_t = tf.concat([tf.sin(t1), tf.cos(t1)], 2)
    H2_t = tf.concat([tf.sin(t2), tf.cos(t2)], 2)
    H1_x = tf.concat([tf.sin(x1), tf.cos(x1)], 2)
    for l in range(len(layers) - 2):
        W = trainable[2 * l]
        b = trainable[2 * l + 1]
        H1_t = tf.tanh(H1_t @ W + b)
        H2_t = tf.tanh(H2_t @ W + b)
        H1_x = tf.tanh(H1_x @ W + b)
    H1 = H1_t * H1_x
    H2 = H2_t * H1_x
    H = tf.concat([H1, H2], 2)
    W = trainable[-2]
    b = trainable[-1]
    f = H @ W + b
    return f[:, :, 0], f[:, :, 1], f[:, :, 2]


def train_step(w1, w2, First, optimizer):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(trainable)
        with tf.GradientTape(persistent=True,
                             watch_accessed_variables=False) as tape2:
            tape2.watch((tw, xw))
            a, u, p = net_u(tw, xw)
        idx = tf.argsort(a)[:, ::10]
        p_sort = tf.experimental.numpy.take_along_axis(p, idx, axis=1)
        a_sort = tf.experimental.numpy.take_along_axis(a, idx, axis=1)
        adiff = tf.experimental.numpy.diff(a_sort[:, ::60], axis=1) + 1e-9
        adiffn = adiff / K.min(adiff)
        pdiff1 = tf.experimental.numpy.diff(p_sort[:, ::60], axis=1)
        pdiff = tf.experimental.numpy.diff(p_sort, axis=1)
        dPdA = pdiff1 / adiffn
        ddPdA = tf.experimental.numpy.diff(dPdA, axis=1)
        P0 = p[:, 0]
        p_pred = p - tf.transpose(P0[None, :])
        P = K.min([
            K.mean(K.abs(p_sort[:, -1] - p_sort[:, 1])),
            (K.mean(K.abs(p_sort[:, -10] - p_sort[:, 10])))
        ]) + 1e-4
        Perror_l = p_sort[:, 1] - p_sort[:, -1]
        Perror = K.sum(Perror_l[Perror_l > 0])
        aRp1p = tf.experimental.numpy.sum(K.abs(pdiff[pdiff < 0])) / (
            P * 300 * 16 * w2) + K.max(K.abs(pdiff[pdiff < 0])) / (P * w2)
        aRp1pp = tf.experimental.numpy.sum(K.abs(
            (ddPdA[ddPdA < 0]))) / (P * w2 * 0.1)
        Lm = (aRp1pp + aRp1p) / 100 + Perror

        La1 = K.square(a_data - a)
        Lu1 = K.square(u_data - u)
        At = tape2.gradient(a, tw)
        Px = tape2.gradient(p, xw)
        ux, ut = tape2.gradient(u, (xw, tw))
        Lrt = K.square(At / stdt + (
            (a[:, 0:1] + amean) * ux) / stdx) + K.square(ut / stdt + Px / stdx)
        loss0 = w1 * K.mean(Lrt) + K.mean(La1) + K.max(
            La1) + K.mean(Lu1) / 25 + Lm
        if First:
            af, uf, pf = net_u(twf, xwf)
            Lp1 = K.square(p_dataf - pf)
            loss = loss0 + K.mean(Lp1)
        else:
            loss = loss0
    grads_u = tape.gradient(loss, trainable)
    optimizer.apply_gradients(zip(grads_u, trainable))
    return loss, (a, u, p)


def plot(field, path):
    plt.imshow(field, extent=(t0, t1, x0, x1), aspect="auto", origin="lower")
    plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    plt.savefig(path)
    plt.close()


plt.rcParams['image.cmap'] = 'jet'
tf.random.set_seed(12345)
tf.keras.backend.set_floatx("float32")
float_type = tf.float32

data = scipy.io.loadmat(sys.argv[1])
for key in "matrix_var", "STPWIP", "PWIP_output":
    if key in data:
        data = data[key]
        break
else:
    sys.stderr.write("train.cse.py: no data in file '%s'\n" % path)

iterations_1 = int(sys.argv[2])
iterations_2 = int(sys.argv[3])
period = int(sys.argv[4])

ST = data[::8, :, :]
A = ST[:, :, 0]
u = ST[:, :, 1]
P = ST[:, :, 2]
t = ST[:, :, 3]
x = ST[:, :, 4]

Pf = data[:, :, 2]
tf0 = data[:, :, 3]
xf = data[:, :, 4]

Us = 5
As = np.mean((np.max(ST[:, :, 0], axis=1) - np.min(ST[:, :, 0], axis=1))) * 10
Xs = np.sqrt(As)
Ts = Xs / Us
Ps = 1060 * (Us**2)

tmean = (np.min(t) + np.max(t)) / (2 * Ts)
stdt = np.std(t) / Ts
tw = (t / Ts - tmean) / stdt
twf = (tf0 / Ts - tmean) / stdt
tw = tf.constant(tw, float_type)
twf = tf.constant(twf, float_type)

xmean = (np.min(x) + np.max(x)) / (2 * Xs)
stdx = np.std(x) / Xs
xw = (x / Xs - xmean) / stdx
xwf = (xf / Xs - xmean) / stdx
xw = tf.constant(xw, float_type)
xwf = tf.constant(xwf, float_type)

umean = np.mean(u) / Us
u_data = u / Us - umean

amean = np.mean(A) / As
a_data = A / As - amean

p_dataf = Pf / Ps

layers = 64, 64, 64, 3
W1_t = tf.random.normal([layers[0] // 2], dtype=float_type)
W2_t = 10 * tf.random.normal([layers[0] // 2], dtype=float_type)
W1_x = tf.random.normal([layers[0] // 2], dtype=float_type)
trainable = []
for l in range(len(layers) - 2):
    W = xavier_init(layers[l], layers[l + 1])
    b = tf.Variable(tf.random.normal([1, layers[l + 1]], dtype=float_type))
    trainable.extend((W, b))
W = xavier_init(2 * layers[-2], layers[-1])
b = tf.Variable(tf.random.normal([1, layers[-1]], dtype=float_type),
                dtype=float_type)
trainable.extend((W, b))
lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-3,
                                                    decay_steps=1000,
                                                    decay_rate=0.9)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
step = tf.function(train_step)
for epoch in range(iterations_1):
    loss, (a, u, p) = step(0.01, 10, True, opt)
    if epoch % period == 0:
        sys.stdout.write("0 %08d %10.2e\n" % (epoch, loss))
lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-4,
                                                    decay_steps=1000,
                                                    decay_rate=0.9)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
step = tf.function(train_step)
for epoch in range(iterations_2):
    loss, (a, u, p) = step(10, 100, False, opt)
    if epoch % period == 0:
        sys.stdout.write("1 %08d %10.2e\n" % (epoch, loss))
t0, t1 = tf0[0, 0], tf0[0, -1]
x0, x1 = xf[0, 0], xf[-1, 0]

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(xwf)
    af, uf, pf = net_u(twf, xwf)
ax = tape.gradient(af, xwf)
area = (af + amean) * As
flow = (uf + umean) * Us
pressure = pf * Ps
plot(data[:, :, 0], "train.a0.png")
plot(area, "train.a1.png")
plot(data[:, :, 1], "train.u0.png")
plot(flow, "train.u1.png")
plot(data[:, :, 2], "train.p0.png")
plot(pressure, "train.p1.png")
plot(ax, "train.ax.png")
plt.plot(tf.experimental.numpy.flatten(area),
         tf.experimental.numpy.flatten(pressure), "x")
plt.savefig("train.pa.png")
plt.close()
dp = np.max(pf, axis=1) - np.min(pf, axis=1)
da = np.max(af, axis=1) - np.min(af, axis=1)
plt.plot(xf[:, 0], da / dp * As / Ps, 'o-')
plt.savefig("train.kp.png")
plt.close()
output = np.empty_like(data)[:, :, :4]
output[:, :, 0] = area
output[:, :, 1] = flow
output[:, :, 2] = pressure
output[:, :, 3] = ax

scipy.io.savemat("output.mat", {"PWIP_output": output})
