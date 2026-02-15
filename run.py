import os
import json
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
import time
from PIL import Image

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from text_extraction import Config, Predictor
from rotation import model, Craft
from graph_predict import PredictGrapher


config = {
    "input": "data_test", # 1 ảnh hoặc 1 folder đều được
    "output": "result",
    "vietocr_model": "vgg_seq2seq",
    "gpu": -1,
    "save_image": True,
    "save_text": True,
    "incline": True
}

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.text_detector = None
        self.text_extractor = None

    def load_image(self, image_path):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def prepare_model(self):
        ocr_config = Config.load_config_from_name(self.config['vietocr_model'])
        if (self.config['gpu'] != 0) and torch.cuda.is_available():
            self.text_detector = Craft('cuda')
            ocr_config['device'] = 'cuda' if self.config['gpu']==-1 else f'cuda:{self.config["gpu"]-1}'
        else:
            self.text_detector = Craft('cpu')
            ocr_config['device'] = 'cpu'
        self.text_extractor = Predictor(ocr_config)

    def rotate(self, img_data):
        img = model.loadImage(img_data['image'])
        bboxes = self.text_detector(img)
        img_data['image'] = img
        img_data['bboxes'] = bboxes
        return img_data

    def extract_info(self, img_data):
        img_data['information'] = []
        incline = {'prev_height': 0, 'prev_line': -1}
        for box in img_data['bboxes']:
            x1 = int(min(box[0][0], box[3][0]))
            y1 = int(min(box[0][1], box[1][1]))
            x2 = int(max(box[2][0], box[1][0]))
            y2 = int(max(box[2][1], box[3][1]))

            arr_img = img_data['image'][y1:y2, x1:x2]
            if arr_img.size==0:
                continue

            try:
                img_box = Image.fromarray(arr_img)
                detected = self.text_extractor.predict(img_box)
            except:
                continue

            if self.config['incline']:
                current_height = sum(y[1] for y in box)/4
                per_diff = abs(1-incline['prev_height']/current_height) if current_height != 0 else 1
                if per_diff < 0.02:
                    img_data['information'][incline['prev_line']].append((box, detected))
                else:
                    img_data['information'].append([(box, detected)])
                    incline['prev_line'] += 1
                incline['prev_height'] = current_height
            else:
                img_data['information'].append([(box, detected)])
        return img_data

    def save_image(self, img_data):
        folder = os.path.join(self.config['output'], "prepare")
        os.makedirs(folder, exist_ok=True)
        img = img_data['image']
        if len(img.shape)==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        path = os.path.join(folder, f"{img_data['name']}.jpg")
        Image.fromarray(img).save(path)

    def save_text(self, img_data):
        folder = os.path.join(self.config['output'], "prepare")
        os.makedirs(folder, exist_ok=True)
        if not img_data.get('information'):
            print(f"[Warning] No information found for {img_data['name']}")
            return
        csv_path = os.path.join(folder, f"{img_data['name']}.csv")
        import csv
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for line in img_data['information']:
                for box, transcript in line:
                    coords = [int(x) for point in box for x in point]
                    writer.writerow(coords + [str(transcript)])
        return csv_path


pl = Pipeline(config)
pl.prepare_model()

input_path = config['input']

if os.path.isfile(input_path):
    image_files = [input_path]
elif os.path.isdir(input_path):
    image_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
else:
    raise ValueError("Input path not found")

for img_path in image_files:

    start_time = time.time()

    img_data = {
        "name": os.path.splitext(os.path.basename(img_path))[0],
        "image": pl.load_image(img_path)
    }

    if img_data["image"] is None:
        print(f"Cannot read image: {img_path}")
        continue

    img_data = pl.rotate(img_data)
    img_data = pl.extract_info(img_data)
    pl.save_image(img_data)
    csv_file = pl.save_text(img_data)
    img_file = os.path.join(config['output'], "prepare", f"{img_data['name']}.jpg")

    end_time = time.time()
    timeocr = end_time - start_time
    print(f"OCR time: {timeocr:.2f} seconds")

    if not csv_file:
        continue

    start_time = time.time()
    model_dir = "models"

    with open(os.path.join(model_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    inv_label_map = {int(v): k for k, v in label_map.items()}
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vec.joblib"))

    grapher = PredictGrapher(csv_file, img_file)
    G, result, df = grapher.graph_formation(export_graph=False)
    df = grapher.relative_distance()

    numeric_cols = ['n_upper','n_alpha','n_spaces','n_numeric','n_special','rd_r','rd_l','rd_t','rd_b']
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0.0
    numeric_feats = df[numeric_cols].astype(float).values
    texts = df['Object'].astype(str).tolist()
    text_feats = tfidf.transform(texts).toarray()
    feats = np.hstack([numeric_feats, text_feats]).astype("float32")
    X = torch.tensor(feats, dtype=torch.float)
    in_feats = X.shape[1]

    edges = []
    for src, dests in result.items():
        for dst in dests:
            edges.append([int(src), int(dst)])
    if len(edges)==0:
        print("No edges found for this file!")
        continue

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=X, edge_index=edge_index)

    num_classes = len(label_map)

    class GNN(nn.Module):
        def __init__(self, in_feats, hidden, num_classes):
            super(GNN, self).__init__()
            self.conv1 = SAGEConv(in_feats, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.fc = nn.Linear(hidden, num_classes)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.fc(x)
            return x

    gnn_model = GNN(in_feats=in_feats, hidden=128, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model.load_state_dict(torch.load(os.path.join(model_dir,"gnn_model.pth"), map_location=device))
    gnn_model.to(device)
    data = data.to(device)
    gnn_model.eval()

    with torch.no_grad():
        logits = gnn_model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)

    df['pred_label'] = [inv_label_map[int(p)] for p in preds]

    os.makedirs("result", exist_ok=True)
    file_name = os.path.splitext(os.path.basename(csv_file))[0]
    out_csv = os.path.join("result", f"{file_name}_predict.csv")
    df[['xmin','ymin','xmax','ymax','Object','pred_label']].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to {out_csv}")

    if os.path.exists(img_file):
        img = cv2.imread(img_file)
        for i, row in df.iterrows():
            label = str(row['pred_label']).strip().lower()
            if label == "other":
                continue
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            text = f"({row['pred_label']})"
            cv2.putText(img, text, (xmin, max(0, ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
        out_img = os.path.join("result", f"{file_name}_predict.jpg")
        cv2.imwrite(out_img, img)
        print(f"Annotated image saved to {out_img}")

    end_time = time.time()
    timegnn = end_time - start_time
    print(f"GNN time: {timegnn:.2f} seconds")
    print(f"Total prediction time: {timeocr + timegnn:.2f} seconds")
