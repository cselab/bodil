#!/usr/bin/env python3

import argparse
import math
import sys
import os


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


parser = argparse.ArgumentParser(
    description=("Train a PINN model on ST/PWIP data.\n\n"
                 "Example usage:\n"
                 "  python train.cse.py \\\n"
                 "    -m data/PWIP/patientData/stenosis1.mat \\\n"
                 "    --i1 10000 --i2 10000 -p 1000 -o out/prediction.mat"),
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--matfile",
                    "-m",
                    required=True,
                    type=str,
                    help="Path to the input .mat file")
parser.add_argument("--i1",
                    required=True,
                    type=int,
                    help="Number of iterations for phase 1 training")
parser.add_argument("--i2",
                    required=True,
                    type=int,
                    help="Number of iterations for phase 2 training")
parser.add_argument("--period",
                    "-p",
                    required=True,
                    type=int,
                    help="Logging and output saving period")
parser.add_argument("--output",
                    "-o",
                    required=True,
                    type=str,
                    help="Output directory prefix")
parser.add_argument("-v",
                    "--verbose",
                    action="store_true",
                    help="Enable verbose output during training")
args = parser.parse_args()
output_dir = os.path.dirname(args.output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
import numpy as np

plt.rcParams['image.cmap'] = 'jet'
tf.random.set_seed(12345)
tf.keras.backend.set_floatx("float32")
float_type = tf.float32

data = scipy.io.loadmat(args.matfile)
for key in "matrix_var", "STPWIP", "PWIP_output":
    if key in data:
        data = data[key]
        break
else:
    sys.stderr.write("train.cse.py: no data in file '%s'\n" % path)

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
for epoch in range(args.i1):
    loss, (a, u, p) = step(0.01, 10, True, opt)
    if args.verbose and epoch % args.period == 0:
        sys.stdout.write("0 %08d %10.4e\n" % (epoch, loss))
lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-4,
                                                    decay_steps=1000,
                                                    decay_rate=0.9)
opt = tf.keras.optimizers.Adam(learning_rate=lr)
step = tf.function(train_step)
for epoch in range(args.i2):
    loss, (a, u, p) = step(10, 100, False, opt)
    if args.verbose and epoch % args.period == 0:
        sys.stdout.write("1 %08d %10.4e\n" % (epoch, loss))
t0, t1 = tf0[0, 0], tf0[0, -1]
x0, x1 = xf[0, 0], xf[-1, 0]

with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    tape.watch(xwf)
    af, uf, pf = net_u(twf, xwf)
ax = tape.gradient(af, xwf)
area = (af + amean) * As
flow = (uf + umean) * Us
pressure = pf * Ps
plot(data[:, :, 0], args.output + "a0.png")
plot(area, args.output + "a1.png")
plot(data[:, :, 1], args.output + "u0.png")
plot(flow, args.output + "u1.png")
plot(data[:, :, 2], args.output + "p0.png")
plot(pressure, args.output + "p1.png")
plot(ax, args.output + "ax.png")
plt.plot(tf.experimental.numpy.flatten(area),
         tf.experimental.numpy.flatten(pressure), "x")
plt.savefig(args.output + "pa.png")
plt.close()
dp = np.max(pf, axis=1) - np.min(pf, axis=1)
da = np.max(af, axis=1) - np.min(af, axis=1)
kp = da / dp * As / Ps
plt.plot(xf[:, 0], kp, 'o-')
plt.savefig(args.output + "kp.png")
plt.close()
with open(args.output + "kp.dat", "w") as file:
    for x0, kp0 in zip(xf[:, 0], kp):
        file.write("%.16e %.16e\n" % (x0, kp0))
output = np.empty_like(data)[:, :, :4]
output[:, :, 0] = area
output[:, :, 1] = flow
output[:, :, 2] = pressure
output[:, :, 3] = ax

scipy.io.savemat(args.output + "output.mat", {"PWIP_output": output})
