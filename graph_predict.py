import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt 
import itertools
import networkx as nx

class PredictGrapher:
    """
    Dùng cho file OCR mới (chưa có label).
    Nhận: file CSV (tọa độ + text) + file ảnh gốc
    Trả về:
        - G: graph (networkx)
        - result: dict {node: [neighbors]}
        - df: dataframe đã có features (không có label)
    """
    def __init__(self, csv_path, img_path):
        self.csv_path = csv_path
        self.img_path = img_path

        # đọc file CSV (OCR output)
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.df = pd.DataFrame(lines)  # mỗi dòng là 1 row

        # đọc ảnh
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        self.image = cv2.imread(img_path)

    def graph_formation(self, export_graph=False):
        df, image = self.df, self.image

        # parse cột
        df = df[0].str.split(',', expand=True)
        temp = df.copy()
        temp[temp.columns] = temp.apply(lambda x: x.str.strip())
        temp.fillna('', inplace=True)

        # gộp text lại (từ cột 8 trở đi)
        temp[8] = temp[8].str.cat(temp.iloc[:,9:], sep=", ")
        temp[temp.columns] = temp.apply(lambda x: x.str.rstrip(", ,"))
        temp = temp.loc[:, :8]

        # bỏ cột thừa
        temp.drop([2,3,6,7], axis=1, inplace=True)
        temp.columns = ['xmin','ymin','xmax','ymax','Object']
        temp[['xmin','ymin','xmax','ymax']] = temp[['xmin','ymin','xmax','ymax']].apply(pd.to_numeric)
        df = temp

        # làm sạch
        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass
        df.dropna(inplace=True)
        df.sort_values(by=['ymin'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["ymax"] = df["ymax"].apply(lambda x: x - 1)

        # group theo dòng (dựa vào tọa độ y)
        master = []
        for idx, row in df.iterrows():
            flat_master = list(itertools.chain(*master))
            if idx not in flat_master:
                top_a = row['ymin']
                bottom_a = row['ymax']
                line = [idx]
                for idx_2, row_2 in df.iterrows():
                    if idx_2 not in flat_master and idx != idx_2:
                        top_b = row_2['ymin']
                        bottom_b = row_2['ymax']
                        if (top_a <= bottom_b) and (bottom_a >= top_b):
                            line.append(idx_2)
                master.append(line)

        df2 = pd.DataFrame({'words_indices': master, 'line_number':[x for x in range(1,len(master)+1)]})
        df2 = df2.set_index('line_number').words_indices.apply(pd.Series).stack().reset_index(level=0).rename(columns={0:'words_indices'})
        df2['words_indices'] = df2['words_indices'].astype('int')
        final = df.merge(df2, left_on=df.index, right_on='words_indices')
        final.drop('words_indices', axis=1, inplace=True)

        final2 = final.sort_values(by=['line_number','xmin'],ascending=True).groupby('line_number').head(len(final)).reset_index(drop=True)
        df = final2

        # tạo graph connections (trái, phải, trên, dưới)
        df.reset_index(inplace=True)
        grouped = df.groupby('line_number')
        horizontal_connections = {}
        left_connections, right_connections = {}, {}

        for _,group in grouped:
            a = group['index'].tolist()
            b = group['index'].tolist()
            horizontal_connection = {a[i]:a[i+1] for i in range(len(a)-1)}
            right_dict_temp = {a[i]:{'right':a[i+1]} for i in range(len(a)-1)}
            left_dict_temp = {b[i+1]:{'left':b[i]} for i in range(len(b)-1)}

            for i in range(len(a)-1):
                df.loc[df['index'] == a[i], 'right'] = int(a[i+1])
                df.loc[df['index'] == a[i+1], 'left'] = int(a[i])

            left_connections.update(right_dict_temp)
            right_connections.update(left_dict_temp)
            horizontal_connections.update(horizontal_connection)

        bottom_connections, top_connections = {}, {}
        for idx, row in df.iterrows():
            if idx not in bottom_connections.keys():
                right_a, left_a = row['xmax'], row['xmin']
                for idx_2, row_2 in df.iterrows():
                    if idx_2 not in bottom_connections.values() and idx < idx_2:
                        right_b, left_b = row_2['xmax'], row_2['xmin']
                        if (left_b <= right_a) and (right_b >= left_a):
                            bottom_connections[idx] = idx_2                
                            top_connections[idx_2] = idx
                            df.loc[df['index'] == idx , 'bottom'] = idx_2
                            df.loc[df['index'] == idx_2, 'top'] = idx 
                            break 

        result = {}
        dic1, dic2 = horizontal_connections, bottom_connections
        for key in (dic1.keys() | dic2.keys()):
            if key in dic1: result.setdefault(key, []).append(dic1[key])
            if key in dic2: result.setdefault(key, []).append(dic2[key])

        G = nx.from_dict_of_lists(result)

        if export_graph:
            if not os.path.exists('graphs'):
                os.makedirs('graphs')
            plot_path = os.path.join('graphs', os.path.basename(self.csv_path) + '_graph.jpg')
            layout = nx.spring_layout(G)
            nx.draw(G, layout, with_labels=True)
            plt.savefig(plot_path, format="PNG", dpi=600)

        self.df = df
        return G, result, df

    def get_text_features(self, df):
        special_chars = ['&', '@', '#', '(',')','-','+', '=', '*', '%', '.', ',', '\\','/', '|', ':']
        n_upper, n_alpha, n_spaces, n_numeric, n_special = [],[],[],[],[]
        for words in df['Object'].tolist():
            upper, alpha, spaces, numeric, special = 0,0,0,0,0
            for char in words:
                if char.isupper(): upper += 1
                if char.isalpha(): alpha += 1
                if char.isspace(): spaces += 1
                if char.isnumeric(): numeric += 1
                if char in special_chars: special += 1
            n_upper.append(upper); n_alpha.append(alpha); n_spaces.append(spaces)
            n_numeric.append(numeric); n_special.append(special)
        df['n_upper'],df['n_alpha'],df['n_spaces'],df['n_numeric'],df['n_special'] = n_upper,n_alpha,n_spaces,n_numeric,n_special

    def relative_distance(self):
        df, img = self.df, self.image
        image_height, image_width = img.shape[0], img.shape[1]
        for index in df['index'].to_list():
            right_index = df.loc[df['index'] == index, 'right'].values[0]
            left_index = df.loc[df['index'] == index, 'left'].values[0]
            bottom_index = df.loc[df['index'] == index, 'bottom'].values[0]
            top_index = df.loc[df['index'] == index, 'top'].values[0]
            if not np.isnan(right_index):
                right_word_left = df.loc[df['index'] == right_index, 'xmin'].values[0]
                source_word_right = df.loc[df['index'] == index, 'xmax'].values[0]
                df.loc[df['index'] == index, 'rd_r'] = (right_word_left - source_word_right)/image_width
            if not np.isnan(left_index):
                left_word_right = df.loc[df['index'] == left_index, 'xmax'].values[0]
                source_word_left = df.loc[df['index'] == index, 'xmin'].values[0]
                df.loc[df['index'] == index, 'rd_l'] = (left_word_right - source_word_left)/image_width
            if not np.isnan(bottom_index):
                bottom_word_top = df.loc[df['index'] == bottom_index, 'ymin'].values[0]
                source_word_bottom = df.loc[df['index'] == index, 'ymax'].values[0]
                df.loc[df['index'] == index, 'rd_b'] = (bottom_word_top - source_word_bottom)/image_height
            if not np.isnan(top_index):
                top_word_bottom = df.loc[df['index'] == top_index, 'ymax'].values[0]
                source_word_top = df.loc[df['index'] == index, 'ymin'].values[0]
                df.loc[df['index'] == index, 'rd_t'] = (top_word_bottom - source_word_top)/image_height
        df[['rd_r','rd_b','rd_l','rd_t']] = df[['rd_r','rd_b','rd_l','rd_t']].fillna(0)
        self.get_text_features(df)
        return df
