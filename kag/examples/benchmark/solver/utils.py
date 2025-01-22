# -*- coding: utf-8 -*-
import os
import json
import pandas as pd


def process_wtq_corpus():

    corpus_path = "data/src/source/WTQ"
    questions_path = "data/src/data/WTQ/"
    questions_file = os.path.join(questions_path, "test.json")
    questions = json.load(open(questions_file, "r"))

    questions_with_source = []
    for item in questions:
        id = item["id"]
        question = item["question"]
        source = item["source"]
        source_file = os.path.join(corpus_path, source).replace(".csv", ".tsv")
        data = pd.read_csv(source_file, sep="\t")
        rows = []
        for row in data.iterrows():
            rows.append(row[1].to_dict())
        item["support_doc"] = rows
        questions_with_source.append(item)

    with open(os.path.join(questions_path, "test_full.json"), "w") as writer:
        writer.write(json.dumps(questions_with_source, indent=4, ensure_ascii=False))
