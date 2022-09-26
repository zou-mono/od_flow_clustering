import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import time
import csv
import os
import sys
import click
import logging
import colorlog
import traceback
import numba as nb
from itertools import chain

# k = 20
# precision = 6  # 判断两个点是否相同的精度，例如6表示小数点后六位相同的坐标值则认为是同一个点

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

log_colors_config = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}

@click.command()
@click.option(
    '--inpath', '-i',
    help='Input table file. For example, d:/res/data/xxx.csv',
    type=str,
    required=True)
@click.option(
    '--k', '-k',
    help='K-nearest-neighborhoods',
    type=int,
    required=True)
@click.option(
    '--precision', '-p',
    help='The precision of coordinate to check if two points are duplicated',
    type=int,
    default=6,
    required=False)
def main(inpath, k, precision):
    # rng = np.random.RandomState(0)
    # X = rng.random_sample((10, 3))
    # tree = KDTree(X, leaf_size=2)
    # dist, ind = tree.query(X[:1], k=3)
    try:
        start = time.time()

        global output_flow_file
        global output_point_file
        global output_mergeflow_file
        global log_file
        global _precision
        global _k
        _precision = precision
        _k = k

        output_flow_file, output_point_file, output_mergeflow_file = check_and_create_outpath()

        global input_data

        log.info("开始计算{}, 参数k={}, p={}".format(inpath, _k, precision))

        log.info("读取数据...")
        input_data = pd.read_excel(inpath)
        # input_data['label'] = input_data.index

        global bWeight
        columns = [col.lower() for col in input_data]
        bWeight = True if 'weight' in columns else False  # 判断是否存在权重字段weight，如果存在则要计算class的权重之和

        O_points = input_data.drop_duplicates(subset='Oid', ignore_index=True)
        D_points = input_data.drop_duplicates(subset='Did', ignore_index=True)

        log.info("生成KD树...")
        # 对O和D分别构建KD树
        global tree_O
        # tree_O = KDTree(input_data[['Ox','Oy']], leaf_size=2)
        tree_O = KDTree(O_points[['Ox','Oy']], leaf_size=2)
        global tree_D
        # tree_D = KDTree(input_data[['Dx','Dy']], leaf_size=2)
        tree_D = KDTree(D_points[['Dx','Dy']], leaf_size=2)
        # dist_o, ind_o = tree_O.query(data[['Ox','Oy']][11896:], k=7)
        # dist_d, ind_d = tree_D.query(data[['Dx','Dy']][11896:], k=7)

        # irow = 0
        # www = []
        # for row in data.itertuples():
        #     closest_O = tree_O.query(np.array([[row.Ox, row.Oy]]), k, return_distance=False)
        #     closest_D = tree_D.query(np.array([[row.Dx, row.Dy]]), k, return_distance=False)
        #
        #     www.append(list(set(closest_O[0]) & set(closest_D[0])))
        #
        #     irow = irow + 1

        # www = data.apply(row_knn, axis=1)
        # data['Oid'] = data['Oid'].apply(lambda column: column + 1)
        # m = data['Oid'].apply(lambda row: row+10)

        # row_knn(data['Ox'], data['Oy'], data['Dx'], data['Dy'])
        log.info("按照定义1.1和定义2计算point和flow的KNN...")

        global df_knn
        df_knn = row_knn(input_data[['Ox', 'Oy']], input_data[['Dx', 'Dy']], O_points, D_points)

        # contiguous_flow_pairs = pd.DataFrame(columns=['p','q', 'dist'])
        global flow_dict
        flow_dict = input_data.to_dict('index')
        merge_class = {}  # 用于存储已经合并过的分类,以及每个class的centroid坐标

        # class_arr = input_data['label'].to_numpy().reshape(-1, 1)
        class_arr = input_data.index.array.to_numpy().astype(np.uint32)  # 用一个array存储flow所在的classID
        flow_arr = input_data.index.array.to_numpy().reshape(-1, 1).tolist()  # 用一个List存储class所包含的flow，可以提高更新class_arr的速度

        del input_data

        log.info("Agglomerative Flow Clustering...")
        iflow_row = 0  # 记录flow的行号
        total_count = len(df_knn)
        iprogress = 0 # 进程百分比
        iprop = 1

        for p_flow in df_knn.itertuples(index=False):
            flow_knn = p_flow.flow_knn

            contiguous_flow_pairs = []

            # 根据dist进行排序
            for contiguous_flow_ID in flow_knn:
                # if len(contiguous_flow_pairs) == _k:  # knn自身也会被算进去，要排除掉
                #     break

                if iflow_row != contiguous_flow_ID:
                    q_flow = df_knn.iloc[contiguous_flow_ID]

                    dist = SNN_flow_distance(p_flow, p_flow, q_flow, q_flow)
                    # contiguous_flow_pairs.loc[i] = [irow, intersect, dist]
                    if dist < 1:
                        contiguous_flow_pairs.append([contiguous_flow_ID, dist])
            # list(map(lambda n: SNN_list(n, iflow_row, p_flow, contiguous_flow_pairs), flow_knn))

            if len(contiguous_flow_pairs) > 1:
                contiguous_flow_pairs.sort(key=lambda x: x[1])

            p_ID = iflow_row
            # 开始聚类
            for flow_pair in contiguous_flow_pairs:
                # p_ID = flow_pair[0]
                q_ID = flow_pair[0]
                dist = flow_pair[1]

                # contiguous_flow_pairs是按照距离升序的，如果 dist==1 那么后面所有的pairs都不邻接，则不需要再向下运算直接跳出
                # 只有 dist < 1 时才向下运算
                # Cx_ID = flow_dict[p_ID]['label']
                # Cy_ID = flow_dict[q_ID]['label']
                Cx_ID = class_arr.item(p_ID)
                Cy_ID = class_arr.item(q_ID)

                if Cx_ID != Cy_ID:
                    # C_ID = Cy_ID if Cx_ID > Cy_ID else Cx_ID # 取比较小的那个class的ID做为合并后class的ID
                    # change_ID = Cy_ID if Cy_ID > Cx_ID else Cx_ID
                    if Cx_ID > Cy_ID:
                        change_ID = Cx_ID
                        C_ID = Cy_ID
                    else:
                        change_ID = Cy_ID
                        C_ID = Cx_ID

                    # if C_ID == 2:
                    #     print("error")
                    # start = time.time()
                    # # class_arr = np.where(class_arr == change_ID, C_ID, class_arr)
                    # class_arr = np.where(class_arr == change_ID, C_ID, class_arr)
                    # end = time.time()
                    # print("np.where {}".format((end-start)*1000000))

                    # start = time.time()
                    # class_arr = assign_value(class_arr, change_ID, C_ID)
                    # end = time.time()
                    # print("numba {}".format((end-start)*1000000))

                    #  如果两条flow不是第一次被访问到，则更新class_dist
                    if Cx_ID == p_ID and Cy_ID == q_ID:
                        class_dist = dist
                    else:
                        class_dist = class_distance(p_ID, q_ID, Cx_ID, Cy_ID, merge_class)

                    if class_dist < 1:
                        if C_ID in merge_class:
                            centroid_O = merge_class[C_ID]['centroid_O']
                            centroid_D = merge_class[C_ID]['centroid_D']
                            weight = merge_class[C_ID]['weight']
                        else:
                            centroid_O = [flow_dict[C_ID]['Ox'], flow_dict[C_ID]['Oy']]
                            centroid_D = [flow_dict[C_ID]['Dx'], flow_dict[C_ID]['Dy']]
                            weight = flow_dict[C_ID]['weight'] if bWeight else 1

                        if change_ID in merge_class:
                            change_centroid_O = merge_class[change_ID]['centroid_O']
                            change_centroid_D = merge_class[change_ID]['centroid_D']
                            change_weight = merge_class[change_ID]['weight']
                        else:
                            change_centroid_O = [flow_dict[change_ID]['Ox'], flow_dict[change_ID]['Oy']]
                            change_centroid_D = [flow_dict[change_ID]['Dx'], flow_dict[change_ID]['Dy']]
                            change_weight = flow_dict[change_ID]['weight'] if bWeight else 1

                        merge_class[C_ID] = {
                            'centroid_O': [round((centroid_O[0] + change_centroid_O[0]) / 2, precision),
                                            round((centroid_O[1] + change_centroid_O[1]) / 2, precision)],
                            'centroid_D': [round((centroid_D[0] + change_centroid_D[0]) / 2, precision),
                                            round((centroid_D[1] + change_centroid_D[1]) / 2, precision)],
                            'weight': weight + change_weight
                        }

                        class_arr = assign_value2_nb(class_arr, np.array(flow_arr[change_ID]), C_ID)
                        flow_arr[C_ID].extend(flow_arr[change_ID])

            if int(iflow_row * 100 / total_count) == iprop * 20:
                log.info("{:.0%}已处理完成...".format(iflow_row / total_count))
                iprop += 1

            iflow_row += 1
            # log.info("{:.0%}".format(iflow_row / total_count))

        log.info("输出聚类结果...")
        output_class(class_arr, merge_class)

        end = time.time()
        log.info("OK. 花费时间{}".format(end-start))

    except FileNotFoundError:
        log.error("输入文件不存在!")
    except ValueError:
        log.error("输入文件格式错误!")
    except:
        log.error(traceback.format_exc())


@nb.jit("uint32[:](uint32[:], int32, int64)", nopython=True)
# @nb.jit(nopython=True)
def assign_value_nb(arr, e, v):
    for i in range(arr.shape[0]):
        if arr[i] == e:
            arr[i] = v
    return arr


def assign_value2(class_arr, update_arr, v):
    for i in update_arr:
        class_arr[i] = v


@nb.jit(nopython=True)
def assign_value2_nb(class_arr, update_arr, v):
    for i in range(update_arr.shape[0]):
        class_arr[update_arr[i]] = v
    return class_arr


def SNN_list(contiguous_flow_ID, iflow_row, p_flow, contiguous_flow_pairs):
    if len(contiguous_flow_pairs) == _k:  # knn自身也会被算进去，要排除掉
        return

    if iflow_row != contiguous_flow_ID:
        q_flow = df_knn.iloc[contiguous_flow_ID]

        dist = SNN_flow_distance(p_flow, p_flow, q_flow, q_flow)
        # contiguous_flow_pairs.loc[i] = [irow, intersect, dist]
        contiguous_flow_pairs.append([contiguous_flow_ID, dist])


def check_and_create_outpath():
    cur_path, filename = os.path.split(os.path.abspath(sys.argv[0]))
    res_path = os.path.join(cur_path, 'res')
    log_path = os.path.join(cur_path, 'logs')

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = os.path.join(log_path, '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))
    output_flow_file = os.path.join(res_path, "{}_flow_k={}.csv".format(time.strftime('%Y-%m-%d-%H-%M-%S'), _k))
    output_point_file = os.path.join(res_path, "{}_point_k={}.csv".format(time.strftime('%Y-%m-%d-%H-%M-%S'), _k))
    output_mergeflow_file = os.path.join(res_path, "{}_mergeflow_k={}.csv".format(time.strftime('%Y-%m-%d-%H-%M-%S'), _k))

    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] [%(levelname)s]- %(message)s',
        log_colors=log_colors_config)  # 日志输出格式
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(filename)s:%(funcName)s:%(lineno)d] [%(levelname)s]- %(message)s')  # 日志输出格式
    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return output_flow_file, output_point_file, output_mergeflow_file


#  将分类结果输出
def output_class(class_arr, merge_class):
    class_arr = np.unique(class_arr)  # 过滤掉重复的class，得到唯一的class ID集合，也就是最终的分类结果

    centroids = {}
    #  给class的中心点重新标定ID号
    for cls in np.nditer(class_arr):
        cID = cls.item()  # class的flow ID号

        #  如果cID在merge_class中，说明该类是合并了flow后生成的，则直接输出merge_class中对应的坐标，id重新命名
        #  否则，cID还是原始的一个flow一个类，id还用原来的
        if cID in merge_class:
            m_cls = merge_class[cID]
            pt_ID = "{}-{}".format(m_cls['centroid_O'][0], m_cls['centroid_O'][1])
            # centroids[pt_ID] = [m_cls['centroid_O'][0], m_cls['centroid_O'][1]]
            centroids[pt_ID] = {
                'id': "o_{}".format(cID),
                'pos': [m_cls['centroid_O'][0], m_cls['centroid_O'][1]]
            }
            pt_ID = "{}-{}".format(m_cls['centroid_D'][0], m_cls['centroid_D'][1])
            # centroids[pt_ID] = [m_cls['centroid_D'][0], m_cls['centroid_D'][1]]
            centroids[pt_ID] = {
                'id': "d_{}".format(cID),
                'pos': [m_cls['centroid_D'][0], m_cls['centroid_D'][1]]
            }
        else:
            m_cls = flow_dict[cID]
            pt_ID = "{}-{}".format(round(m_cls['Ox'], 6), round(m_cls['Oy'], _precision))
            # centroids[pt_ID] = [m_cls['Ox'], m_cls['Oy']]
            centroids[pt_ID] = m_cls['Oid']
            centroids[pt_ID] = {
                'id': m_cls['Oid'],
                'pos': [m_cls['Ox'], m_cls['Oy']]
            }
            pt_ID = "{}-{}".format(round(m_cls['Dx'], 6), round(m_cls['Dy'], _precision))
            # centroids[pt_ID] = [m_cls['Dx'], m_cls['Dy']]
            centroids[pt_ID] = {
                'id': m_cls['Did'],
                'pos': [m_cls['Dx'], m_cls['Dy']]
            }

    # 输出点文件
    # point_dict = {}  # key, value = 坐标值key(以“-”连接), 新ID
    header = ['name', 'lng', 'lat']
    with open(output_point_file, 'w+',  newline='') as o:
        writer = csv.writer(o)
        writer.writerow(header)

        irow = 1
        for key, value in centroids.items():
            row = [value['id'], value['pos'][0], value['pos'][1]]
            # point_dict[key] = newID
            writer.writerow(row)
            irow += 1

    header = ['from', 'to', 'weight']
    o1 = open(output_flow_file, 'w+',  newline='')
    writer_flow = csv.writer(o1)
    o2 = open(output_mergeflow_file, 'w+',  newline='')
    write_mergeflow = csv.writer(o2)

    writer_flow.writerow(header)
    write_mergeflow.writerow(header)

    for cls in np.nditer(class_arr):
        cID = cls.item()  # class的flow ID号

        row = []

        #  如果cID在merge_class中，说明该类是合并了flow后生成的，则直接输出merge_class中对应的坐标
        #  否则，cID还是原始的一个flow一个类，则输出
        if cID in merge_class:
            cls = merge_class[cID]
            pt_ID = "{}-{}".format(cls['centroid_O'][0], cls['centroid_O'][1])
            row.append(centroids[pt_ID]['id'])
            pt_ID = "{}-{}".format(cls['centroid_D'][0], cls['centroid_D'][1])
            row.append(centroids[pt_ID]['id'])
            row.append(cls['weight'])

            write_mergeflow.writerow(row)
        else:
            cls = flow_dict[cID]
            row.append(cls['Oid'])
            row.append(cls['Did'])
            if bWeight:
                row.append(cls['weight'])
            else:
                row.append(1)
        writer_flow.writerow(row)

    o1.close()
    o2.close()

    # print("debug")


#  Shared Nearest Neighbor (SNN) flow distance
def SNN_flow_distance(p_flow_o, p_flow_d, q_flow_o, q_flow_d):
    m = list(set(p_flow_o.O_knn) & set(q_flow_o.O_knn))
    n = list(set(p_flow_d.D_knn) & set(q_flow_d.D_knn))
    dist = 1 - len(m) * len(n) / (_k ** 2)

    return dist


# 两个flow class之间的近似距离
def class_distance(p_ID, q_ID, Cx_ID, Cy_ID, flow_class):
    if Cx_ID != p_ID:
        Cx_centeroids_O = np.array(flow_class[Cx_ID]['centroid_O']).reshape(1, 2)
        Cx_centeroids_D = np.array(flow_class[Cx_ID]['centroid_D']).reshape(1, 2)
    else:
        Cx_centeroids_O = np.array([flow_dict[Cx_ID]['Ox'], flow_dict[Cx_ID]['Oy']]).reshape(1, 2)
        Cx_centeroids_D = np.array([flow_dict[Cx_ID]['Dx'], flow_dict[Cx_ID]['Dy']]).reshape(1, 2)

    if Cy_ID != q_ID:
        Cy_centeroids_O = np.array(flow_class[Cy_ID]['centroid_O']).reshape(1,2)
        Cy_centeroids_D = np.array(flow_class[Cy_ID]['centroid_D']).reshape(1,2)
    else:
        Cy_centeroids_O = np.array([flow_dict[Cy_ID]['Ox'], flow_dict[Cy_ID]['Oy']]).reshape(1, 2)
        Cy_centeroids_D = np.array([flow_dict[Cy_ID]['Dx'], flow_dict[Cy_ID]['Dy']]).reshape(1, 2)

    O_Cx = closest_point(tree_O, Cx_centeroids_O, Cx_ID)
    O_Cy = closest_point(tree_O, Cy_centeroids_O, Cy_ID)

    D_Cx = closest_point(tree_D, Cx_centeroids_D, Cx_ID)
    D_Cy = closest_point(tree_D, Cy_centeroids_D, Cy_ID)

    # 这里还有优化空间，可以考虑将df_knn转换为dict
    dist = SNN_flow_distance(df_knn.iloc[O_Cx], df_knn.iloc[O_Cy], df_knn.iloc[D_Cx], df_knn.iloc[D_Cy])

    return dist

    # print("debug")


def closest_point(tree, query_point, query_no):
    closest_point = tree.query(query_point, 2, return_distance=False)
    if closest_point[0][0] == query_no:
        return closest_point[0][1]
    else:
        return closest_point[0][0]


# def get_OD_ID(flow_ind):
#     return input_data.iloc[flow_ind].Oid, input_data.iloc[flow_ind].Did


#  逐行计算O，D点的KNN，以及flow的KNN
def row_knn(O, D, O_points, D_points):
    #  k近邻实际上求的是k+1近邻，因为会把自身也算进去
    log.info("KD树检索...")
    closest_O = tree_O.query(O, _k + 1, return_distance=False)
    closest_O = closest_O[:,1:]
    # log.info("计算D点的KNN...")
    closest_D = tree_D.query(D, _k + 1, return_distance=False)
    closest_D = closest_D[:,1:]

    df = pd.DataFrame(columns=['O_knn', 'D_knn'])
    OD_id = input_data[['Oid', 'Did']]

    log.info("计算O点KNN...")
    # OD_id = OD_id.to_dict()
    # vfunc_O = np.vectorize(lambda n: OD_id['Oid'][n])
    # vfunc_D = np.vectorize(lambda n: OD_id['Did'][n])
    OD_id = O_points['Oid'].to_dict()
    vfunc_O = np.vectorize(lambda n: OD_id[n])
    closest_pt = vfunc_O(closest_O)
    df['O_knn'] = np.array(closest_pt).tolist()

    log.info("计算D点KNN...")
    OD_id = D_points['Did'].to_dict()
    vfunc_D = np.vectorize(lambda n: OD_id[n])
    closest_pt = vfunc_D(closest_D)
    df['D_knn'] = np.array(closest_pt).tolist()

    # v = np.vectorize(lambda n: np.where(OD_id['Oid'] == n))
    # vv = v(O_points['Oid'].tolist())

    OD_id = input_data[['Oid', 'Did']]

    log.info("计算O点KNN所属的flow...")
    closest_pt = np.array(df['O_knn']).tolist()
    # s = time.time()
    # id_arr = input_data['Oid'].to_numpy()
    O_points = O_points['Oid'].tolist()
    df_flows = pd.DataFrame(columns=['Oid', 'flows'])
    df_flows['Oid'] = O_points
    df_flows['flows'] = list(map(lambda n: np.where(OD_id['Oid'] == n), O_points))
    df_flows = df_flows.set_index('Oid')
    df_dict = df_flows['flows']

    flowIDs = query_flowIDs_by_pointIDs_as_list2(closest_pt, df_dict)
    # flowIDs = query_flowIDs_by_pointIDs_as_list(closest_O, id_arr)
    df['O_flow_knn'] = list(map(lambda m: np.concatenate(m, axis=1).tolist()[0], flowIDs))
    # e = time.time()
    # print(e - s)

    log.info("计算D点KNN所属的flow...")
    # id_arr = input_data['Did'].to_numpy()
    closest_pt = np.array(np.array(df['D_knn'])).tolist()
    D_points = D_points['Did'].tolist()
    df_flows = pd.DataFrame(columns=['Did', 'flows'])
    df_flows['Did'] = D_points
    df_flows['flows'] = list(map(lambda n: np.where(OD_id['Did'] == n), D_points))
    df_flows = df_flows.set_index('Did')
    df_dict = df_flows['flows']

    del df_flows
    del D_points

    # flowIDs = query_flowIDs_by_pointIDs_as_list(closest_D, id_arr)
    flowIDs = query_flowIDs_by_pointIDs_as_list2(closest_pt, df_dict)
    df['D_flow_knn'] = list(map(lambda m: np.concatenate(m, axis=1).tolist()[0], flowIDs))

    # 稍慢
    # s = time.time()
    # df['O_flow_knn'] = \
    #     list(map(
    #         lambda n: input_data[input_data['Oid'].isin(n)].index.tolist(),
    #         np.array(closest_O).tolist()))
    # e = time.time()
    # print(e-s)
    #
    # df['D_flow_knn'] = \
    #     list(map(
    #         lambda n: input_data[input_data['Did'].isin(n)].index.tolist(),
    #         np.array(closest_D).tolist()))

    # df['o_knn'] = closest_O.apply(lambda s: s.to_numpy(), axis=1)
    # df['d_knn'] = closest_D.apply(lambda s: s.to_numpy(), axis=1)

    # 稍慢
    # m = df.apply(intersect, axis=1)

    # 更快
    log.info("计算flow的KNN...")
    df['flow_knn'] = [list(set(a).intersection(set(b))) for a, b in zip(df['O_flow_knn'], df['D_flow_knn'])]

    return df

def query_flowIDs_by_pointIDs_as_list2(closest_point, point_flows_dict):
    return list(map(lambda m: query_flowIDs_by_pointID2(m, point_flows_dict), closest_point))


def query_flowIDs_by_pointID2(pointIDs, point_flows_dict):
    return list(map(lambda m: point_flows_dict[m], pointIDs))


def query_flowIDs_by_pointIDs_as_list(closest_point, id_arr):
    return list(map(lambda m: query_flowIDs_by_pointID(m, id_arr), closest_point))


# 根据point的ID号查询所在flow的ID数组
def query_flowIDs_by_pointID(pointIDs, id_arr):
    return list(map(lambda m: np.where(id_arr == m), pointIDs))

# def intersect(row):
#     return list(set(row['o_knn']) & set(row['d_knn']))


if __name__ == '__main__':
    # start = time.time()
    main()
    # end = time.time()
    # print("OK. 花费时间{}".format(end-start))

