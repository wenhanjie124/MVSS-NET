# coding=utf-8
import time
import pandas as pd
import seaborn as sns
from create_mul_view_record import Read_mul_view_tfrecords, Get_tfrecords_nums
from tools.MVCNN import MVCNN
from tools.utils import *

os.environ["PATH"] += os.pathsep + 'D:/Program Files/graphviz-2.38/release/bin'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse(dataset, ll, rr):
    base_ = []
    val_ = []
    j = -1
    images = []
    a = []
    b = []
    for i, (iamge, lable) in enumerate(dataset):
        if (int(lable.numpy()) < ll) or (int(lable.numpy()) > rr):
            if int(lable.numpy()) != j:
                j = int(lable.numpy())
                images = []
                images.append(iamge.numpy())
                a.append(lable.numpy())
            else:
                images.append(iamge.numpy())
                if images.__len__() == 2:
                    base_.append(images[0])
                    val_.append(images[1])
                    b.append(lable.numpy())
    base_ = np.asarray(base_)
    val_ = np.asarray(val_)
    return base_, val_


def caculatecos(val_feature, base_feature):
    score = np.sum(np.multiply(val_feature, base_feature))
    score = score / (np.sqrt(np.sum(np.square(val_feature))) * np.sqrt((np.sum(np.square(base_feature)))))
    return score


def Normalize(data):
    # m = data.mean()
    print(data.shape)
    mx = data.min()
    mn = data.max()
    print(data.shape)
    for i in range(data.__len__()):
        for j in range(data.__len__()):
            data[i][j] = (data[i][j] - mx) / (mn - mx)
    return data


def heatmap(data=None):
    if data is None:
        fm = pd.read_csv("fm.csv")
        p1 = sns.heatmap(fm, xticklabels=10, yticklabels=10, annot=False, fmt=".1f", vmin=0,
                         vmax=1)  # OrRd,YlOrRd,RdYlGn_r,PuBu
        # p1.xaxis.set_ticks_position("top")
        p1.invert_yaxis()
        plt.title("Heat Map(132)", fontdict={'family': 'Times New Roman',
                                             'style': 'italic',
                                             'weight': 'medium',
                                             'color': 'black',
                                             'size': 20
                                             })
        plt.savefig("heatmap132.png", dpi=600)
        plt.show()
        fm.drop(fm.index[10:], inplace=True)
        fm.drop(fm.columns[10:], axis=1, inplace=True)
        p1 = sns.heatmap(fm, xticklabels=1, yticklabels=1, annot=True, fmt=".2f", vmin=0,
                         vmax=1)  # OrRd,YlOrRd,RdYlGn_r,PuBu
        p1.set_facecolor("none")
        # p1.xaxis.set_ticks_position("top")
        p1.invert_yaxis()
        plt.title("Heat Map(10)", fontdict={'family': 'Times New Roman',
                                            'style': 'italic',
                                            'weight': 'medium',
                                            'color': 'black',
                                            'size': 20
                                            })
        plt.savefig("heatmap10.png", dpi=600)
        plt.show()
        return
    data = pd.DataFrame(data)
    data.to_csv("fm.csv", index=False)
    fm = pd.read_csv("fm.csv")
    p1 = sns.heatmap(fm, xticklabels=10, yticklabels=10, annot=False, fmt=".1f", vmin=0,
                     vmax=1)  # OrRd,YlOrRd,RdYlGn_r,PuBu
    p1.xaxis.set_ticks_position("top")
    plt.title("Heat Map(132)", fontdict={'family': 'Times New Roman',
                                         'style': 'italic',
                                         'weight': 'medium',
                                         'color': 'black',
                                         'size': 20
                                         })
    plt.savefig("heatmap132.png", dpi=600)
    plt.show()
    fm.drop(fm.index[10:], inplace=True)
    fm.drop(fm.columns[10:], axis=1, inplace=True)
    p1 = sns.heatmap(fm, xticklabels=1, yticklabels=1, annot=True, fmt=".2f", vmin=0,
                     vmax=1)  # OrRd,YlOrRd,RdYlGn_r,PuBu
    p1.set_facecolor("none")
    p1.xaxis.set_ticks_position("top")
    plt.title("Heat Map(10)", fontdict={'family': 'Times New Roman',
                                        'style': 'italic',
                                        'weight': 'medium',
                                        'color': 'black',
                                        'size': 20
                                        })
    plt.savefig("heatmap10.png", dpi=600)
    plt.show()
    return


if __name__ == "__main__":
    # heatmap()
    time_be = time.time()
    train_data = Read_mul_view_tfrecords('data/total_500.tfrecords', gray=True)
    train_len = Get_tfrecords_nums('data/total_500.tfrecords')
    mode_name = "modelcheck(4)"
    line = np.loadtxt(f'{mode_name}/left_right.txt', dtype=str, delimiter="\n")
    line = line.tolist()
    ll, rr = line.split()[-2:]
    ll = ll[4:]
    rr = rr[4:]
    ll = int(ll)
    rr = int(rr)

    print(train_len, ll, rr)
    base_data, val_data = parse(train_data, ll, rr)
    print(base_data.shape)
    print(val_data.shape)
    data_len = base_data.shape[0]
    base_model = MVCNN(input_shape=(12, 500, 500, 1), classes=600).build()
    top1_flag = 0
    top5_flag = 0
    for i in os.listdir(mode_name):
        if i.endswith(".hdf5"):
            model_fuck = i
    base_model.load_weights(f'{mode_name}/{model_fuck}')
    print(model_fuck)
    model = keras.Model(inputs=base_model.input, outputs=base_model.get_layer('concatenate_layer').output)
    time_start = time.time()
    base = model.predict(base_data, batch_size=4)
    val = model.predict(val_data, batch_size=4)
    time_end = time.time()
    time1 = (time_end - time_start) / 50
    top1 = 0
    top5 = 0
    val_label = [i for i in range(data_len)]
    base_label = [i for i in range(data_len)]
    feature_map = np.zeros((132, 132), dtype=float)
    print(feature_map.shape)
    time_start = time.time()
    for i, val_feature in enumerate(val):
        ditu = {}
        for j, base_feature in enumerate(base):
            ditu[j] = caculatecos(val_feature, base_feature)
            feature_map[i][j] = ditu[j]
        ditu = sorted(ditu.items(), key=lambda x: x[1], reverse=True)
        for k in range(5):
            if k == 0:
                if val_label[i] == base_label[ditu[k][0]]:
                    top1 += 1
                    top5 += 1
                    # print(val_label[i], ditu[k], ditu[k][0], base_label[ditu[k][0]])
                    break
            if val_label[i] == base_label[ditu[k][0]]:
                top5 += 1
                break
    if top1 > top1_flag:
        top1_flag = top1
        top5_flag = top5
        name = model_fuck
    time_end = time.time()
    time2 = (time_end - time_start) / 132
    print("on average_time ", time1 + time1)
    print('processing model is {}   top1 is {:.2f}%'.format(model_fuck, top1 / data_len))
    print("**" * 10 + 'test finishd' + "**" * 10)
    print('top1 {:.2f}%'.format(top1_flag / data_len * 100))
    print('top5 {:.2f}%'.format(top5_flag / data_len * 100))
    # print(feature_map)
    # feature_map = Normalize(feature_map)
    # time_en = time.time()
    # print(time_en - time_be)
    # print((time_en - time_be) / 132)
    # heatmap(feature_map)
