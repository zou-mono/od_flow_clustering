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
        global log_file
        global _precision
        global _k
        _precision = precision
        _k = k

        output_flow_file, output_point_file = check_and_create_outpath()

        global input_data

        log.info("开始计算{}, 参数k={}, p={}".format(inpath, _k, precision))

        log.info("读取数据...")
        input_data = pd.read_excel(inpath)
        # input_data['label'] = input_data.index

        global bWeight
        columns = [col.lower() for col in input_data]
        bWeight = True if 'weight' in columns else False  # 判断是否存在权重字段weight，如果存在则要计算class的权重之和

        O_points = input_data.drop_duplicates(subset='Oid')
        D_points = input_data.drop_duplicates(subset='Did')

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
        class_arr = input_data.index.array.to_numpy().reshape(-1, 1)

        del input_data

        log.info("Agglomerative Flow Clustering...")
        iflow_row = 0  # 记录flow的行号
        total_count = len(df_knn)
        iprogress = 0 # 进程百分比
        iprop = 1

        for p_flow in df_knn.itertuples():
            flow_knn = p_flow.flow_knn

            contiguous_flow_pairs = []

            # 根据dist进行排序
            for contiguous_flow_ID in flow_knn:
                if len(contiguous_flow_pairs) == k:  # knn自身也会被算进去，要排除掉
                    break

                if iflow_row != contiguous_flow_ID:
                    q_flow = df_knn.iloc[contiguous_flow_ID]

                    dist = SNN_flow_distance(p_flow, p_flow, q_flow, q_flow)
                    # contiguous_flow_pairs.loc[i] = [irow, intersect, dist]
                    contiguous_flow_pairs.append([contiguous_flow_ID, dist])

            if len(contiguous_flow_pairs) > 1:
                contiguous_flow_pairs.sort(key=lambda x: x[1])

            # 开始聚类
            for flow_pair in contiguous_flow_pairs:
                # p_ID = flow_pair[0]
                p_ID = iflow_row
                q_ID = flow_pair[0]
                dist = flow_pair[1]

                # if iflow_row == 5676:
                #     print("error")

                # contiguous_flow_pairs是按照距离升序的，如果 dist==1 那么后面所有的pairs都不邻接，则不需要再向下运算直接跳出
                # 只有 dist < 1 时才向下运算
                if dist < 1:
                    # Cx_ID = flow_dict[p_ID]['label']
                    # Cy_ID = flow_dict[q_ID]['label']
                    Cx_ID = class_arr.item(p_ID)
                    Cy_ID = class_arr.item(q_ID)

                    if Cx_ID != Cy_ID:
                        C_ID = Cy_ID if Cx_ID > Cy_ID else Cx_ID # 取比较小的那个class的ID做为合并后class的ID
                        change_ID = q_ID if C_ID == p_ID else p_ID

                        # if C_ID == 5677:
                        #     print("debug")

                        #  如果两条flow都是第一次被访问到，则更新flow_class后结束本次迭代
                        if Cx_ID == p_ID and Cy_ID == q_ID:
                            # flow_dict[q_ID]['label'] = Cx_ID
                            if bWeight:
                                weight = flow_dict[p_ID]['weight'] + flow_dict[q_ID]['weight']
                            else:
                                weight = 2

                            merge_class[C_ID] = {
                                'centroid_O': [round((flow_dict[Cx_ID]['Ox'] + flow_dict[Cy_ID]['Ox']) / 2, precision),
                                                round((flow_dict[Cx_ID]['Oy'] + flow_dict[Cy_ID]['Oy']) / 2, precision)],
                                'centroid_D': [round((flow_dict[Cx_ID]['Dx'] + flow_dict[Cy_ID]['Dx']) / 2, precision),
                                                round((flow_dict[Cx_ID]['Dy'] + flow_dict[Cy_ID]['Dy']) / 2, precision)],
                                'weight': weight
                            }

                            class_arr = np.where(class_arr == change_ID, C_ID, class_arr)

                            continue
                        #  如果两条flow不是第一次被访问到，则需要重新计算class_distance
                        else:
                            class_dist = class_distance(p_ID, q_ID, Cx_ID, Cy_ID, merge_class)

                            # if Cx_ID != p_ID and Cy_ID != q_ID and class_dist < 1:
                            #     print("error")

                            if class_dist < 1:
                                if Cx_ID != p_ID and Cy_ID == q_ID:
                                    if bWeight:
                                        weight = merge_class[Cx_ID]['weight'] + flow_dict[q_ID]['weight']
                                    else:
                                        weight = merge_class[Cx_ID]['weight'] + 1

                                    merge_class[C_ID] = {
                                        'centroid_O': [round((merge_class[Cx_ID]['centroid_O'][0] + flow_dict[Cy_ID]['Ox']) / 2, precision),
                                                       round((merge_class[Cx_ID]['centroid_O'][1] + flow_dict[Cy_ID]['Oy']) / 2, precision)],
                                        'centroid_D': [round((merge_class[Cx_ID]['centroid_D'][0] + flow_dict[Cy_ID]['Dx']) / 2, precision),
                                                       round((merge_class[Cx_ID]['centroid_D'][1] + flow_dict[Cy_ID]['Dy']) / 2, precision)],
                                        'weight': weight

                                    }
                                    class_arr = np.where(class_arr == change_ID, C_ID, class_arr)

                                elif Cx_ID == p_ID and Cy_ID != q_ID:
                                    if bWeight:
                                        weight = merge_class[Cy_ID]['weight'] + flow_dict[p_ID]['weight']
                                    else:
                                        weight = merge_class[Cy_ID]['weight'] + 1

                                    merge_class[C_ID] = {
                                        'centroid_O': [round((merge_class[Cy_ID]['centroid_O'][0] + flow_dict[Cx_ID]['Ox']) / 2, precision),
                                                       round((merge_class[Cy_ID]['centroid_O'][1] + flow_dict[Cx_ID]['Oy']) / 2, precision)],
                                        'centroid_D': [round((merge_class[Cy_ID]['centroid_D'][0] + flow_dict[Cx_ID]['Dx']) / 2, precision),
                                                       round((merge_class[Cy_ID]['centroid_D'][1] + flow_dict[Cx_ID]['Dy']) / 2, precision)],
                                        'weight': weight

                                    }
                                    class_arr = np.where(class_arr == change_ID, C_ID, class_arr)

                                elif Cx_ID != p_ID and Cy_ID != q_ID:
                                    weight = merge_class[Cx_ID]['weight'] + merge_class[Cy_ID]['weight']

                                    merge_class[C_ID] = {
                                        'centroid_O': [round((merge_class[Cx_ID]['centroid_O'][0] + merge_class[Cy_ID]['centroid_O'][0]) / 2, precision),
                                                       round((merge_class[Cx_ID]['centroid_O'][1] + merge_class[Cy_ID]['centroid_O'][1]) / 2, precision)],
                                        'centroid_D': [round((merge_class[Cx_ID]['centroid_D'][0] + merge_class[Cy_ID]['centroid_D'][0]) / 2, precision),
                                                       round((merge_class[Cx_ID]['centroid_D'][1] + merge_class[Cy_ID]['centroid_D'][1]) / 2, precision)],
                                        'weight': weight
                                    }

                                    class_arr = np.where(class_arr == p_ID, C_ID, class_arr)
                                    class_arr = np.where(class_arr == q_ID, C_ID, class_arr)
                                    if Cx_ID > Cy_ID:
                                        class_arr = np.where(class_arr == Cx_ID, C_ID, class_arr)
                                    else:
                                        class_arr = np.where(class_arr == Cy_ID, C_ID, class_arr)
                else:
                    break

            if int(iflow_row * 100 / total_count) == iprop * 20:
                log.info("{:.0%}已处理完成...".format(iflow_row / total_count))
                iprop += 1

            iflow_row += 1
            # log.info("{:.0%}".format(iflow_row / total_count))

        output_class(class_arr, merge_class)

        end = time.time()
        log.info("OK. 花费时间{}".format(end-start))

    except FileNotFoundError:
        log.error("输入文件不存在!")
    except ValueError:
        log.error("输入文件格式错误!")
    except:
        log.error(traceback.format_exc())


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

    return output_flow_file, output_point_file


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
    with open(output_flow_file, 'w+',  newline='') as o:
        writer = csv.writer(o)
        writer.writerow(header)

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
            else:
                cls = flow_dict[cID]
                row.append(cls['Oid'])
                row.append(cls['Did'])
                if bWeight:
                    row.append(cls['weight'])
                else:
                    row.append(1)
            writer.writerow(row)

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
    # log.info("计算D点的KNN...")
    closest_D = tree_D.query(D, _k + 1, return_distance=False)

    df = pd.DataFrame(columns=['O_knn', 'D_knn'])
    OD_id = input_data[['Oid', 'Did']]

    log.info("计算O点KNN...")
    OD_id = OD_id.to_dict()
    vfunc_O = np.vectorize(lambda n: OD_id['Oid'][n])
    vfunc_D = np.vectorize(lambda n: OD_id['Did'][n])

    log.info("计算D点KNN...")
    closest_pt = vfunc_O(closest_O)
    df['O_knn'] = np.array(closest_pt).tolist()
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
    del df_flows
    del O_points

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
    df['flow_knn'] = [list(set(a).intersection(b)) for a, b in zip(df['O_flow_knn'], df['D_flow_knn'])]

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

