# -*- coding: utf-8 -*-
# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.

import json
import copy
import logging
import pandas as pd
import markdown
from io import StringIO
from typing import Dict, Type, List, Set

from kag.interface import LLMClient
from tenacity import stop_after_attempt, retry

from kag.interface import ExtractorABC, PromptABC, ExternalGraphLoaderABC
from kag.builder.model.chunk import Chunk, ChunkTypeEnum

from kag.common.conf import KAG_PROJECT_CONF
from kag.common.utils import processing_phrases, to_camel_case
from kag.builder.model.chunk import Chunk
from kag.builder.component.extractor.schema_free_extractor import SchemaFreeExtractor
from kag.builder.model.sub_graph import SubGraph
from kag.builder.prompt.utils import init_prompt_with_fallback
from knext.schema.client import OTHER_TYPE, CHUNK_TYPE, BASIC_TYPES
from knext.common.base.runnable import Input, Output
from knext.schema.client import SchemaClient
from kag.builder.model.sub_graph import SubGraph, Node, Edge

from kag.builder.component.table.table_cell import TableCell, TableInfo

logger = logging.getLogger(__name__)


@ExtractorABC.register("table_and_text_extractor")
class TableAndTextExtractor(ExtractorABC):
    """
    A class for extracting knowledge graph subgraphs from text using a large language model (LLM).
    Inherits from the Extractor base class.

    Attributes:
        llm (LLMClient): The large language model client used for text processing.
        schema (SchemaClient): The schema client used to load the schema for the project.
        ner_prompt (PromptABC): The prompt used for named entity recognition.
        std_prompt (PromptABC): The prompt used for named entity standardization.
        triple_prompt (PromptABC): The prompt used for triple extraction.
        external_graph (ExternalGraphLoaderABC): The external graph loader used for additional NER.
    """

    def __init__(
        self,
        llm: LLMClient,
        ner_prompt: PromptABC = None,
        std_prompt: PromptABC = None,
        triple_prompt: PromptABC = None,
        table_classify_prompt: PromptABC = None,
        table_context_prompt: PromptABC = None,
        table_keywords_prompt: PromptABC = None,
        table_reformat_prompt: PromptABC = None,
        external_graph: ExternalGraphLoaderABC = None,
    ):
        """
        Initializes the KAGExtractor with the specified parameters.

        Args:
            llm (LLMClient): The large language model client.
            ner_prompt (PromptABC, optional): The prompt for named entity recognition. Defaults to None.
            std_prompt (PromptABC, optional): The prompt for named entity standardization. Defaults to None.
            triple_prompt (PromptABC, optional): The prompt for triple extraction. Defaults to None.
            external_graph (ExternalGraphLoaderABC, optional): The external graph loader. Defaults to None.
        """
        super().__init__()
        self.schema_free_extractor = SchemaFreeExtractor(
            llm, ner_prompt, std_prompt, triple_prompt, external_graph
        )
        self.llm = llm
        self.table_classify_prompt = table_classify_prompt
        self.table_context_prompt = table_context_prompt
        self.table_keywords_prompt = table_keywords_prompt
        self.table_reformat = table_reformat_prompt

    @property
    def input_types(self) -> Type[Input]:
        return Chunk

    @property
    def output_types(self) -> Type[Output]:
        return SubGraph

    def _invoke(self, input: Input, **kwargs) -> List[Output]:
        """
        Invokes the semantic extractor to process input data.

        Args:
            input (Input): Input data containing name and content.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Output]: A list of processed results, containing subgraph information.
        """
        table_chunk: Chunk = input
        if table_chunk.type == ChunkTypeEnum.Table:
            return self._invoke_table(input, **kwargs)
        # return self.schema_free_extractor._invoke(input, **kwargs)
        return []

    def _invoke_table(self, input: Chunk, **kwargs) -> List[Output]:
        self._table_classify(input)
        return self._table_extractor(input)

    def _table_extractor(self, input: Chunk):
        table_type = input.kwargs["table_type"]

        if table_type in ["指标型表格", "Metric_Based_Table"]:
            return self._extract_metric_table(input)
        elif table_type in ["简单表格", "Simple_Table"]:
            return self._extract_simple_table(input)
        return self._extract_other_table(input)

    def _table_classify(self, input: Chunk):
        # 提取全局信息
        table_desc, keywords, table_name = self._get_table_context(table_chunk=input)
        table_desc = "\n".join(table_desc)

        _content = input.content
        classify_input = {"table": _content, "context": table_desc}

        table_type, table_info = self.llm.invoke(
            {
                "input": json.dumps(classify_input, ensure_ascii=False, sort_keys=True),
            },
            self.table_classify_prompt,
            with_json_parse=True,
            with_except=True,
        )
        input.kwargs["table_type"] = table_type
        input.kwargs["table_info"] = table_info
        input.kwargs["table_name"] = table_name
        input.kwargs["context"] = table_desc
        input.kwargs["keywords"] = keywords
        return input

    def _get_table_context(self, table_chunk: Chunk):
        # 提取表格全局关键字
        table_desc = ""
        keywords = []
        table_context_str = self._get_table_context_str(table_chunk=table_chunk)
        _table_context = self.llm.invoke(
            {
                "input": table_context_str,
            },
            self.table_context_prompt,
            with_json_parse=True,
            with_except=True,
        )
        table_desc = _table_context["table_desc"]
        keywords = _table_context["keywords"]
        table_name = _table_context["table_name"]
        return table_desc, keywords, table_name

    def _get_table_context_str(self, table_chunk: Chunk):
        if "context" in table_chunk.kwargs:
            table_context_str = table_chunk.name + "\n" + table_chunk.kwargs["context"]
        else:
            table_context_str = table_chunk.name + "\n" + table_chunk.content
        if len(table_context_str) <= 0:
            return None
        return table_context_str

    def _extract_metric_table(self, input_table: Chunk):
        table_info = input_table.kwargs["table_info"]
        header = table_info["header"]
        index_col = table_info["index_col"]
        table_df, header, index_col = self._std_table2(input_table=input_table)
        table_name = input_table.kwargs["table_name"]
        cell_value_desc = None
        scale = table_info.get("scale", None)
        units = table_info.get("units", None)
        if scale is not None:
            cell_value_desc = str(scale)
        if units is not None and isinstance(units, str):
            cell_value_desc += "," + units
        if cell_value_desc is not None:
            cell_value_desc = "(" + cell_value_desc + ")"

        table_cell_info: TableInfo = self._generate_table_cell_info(
            data=table_df,
            header=header,
            table_name=table_name,
            cell_value_desc=cell_value_desc,
        )
        table_cell_info.sacle = table_info.get("scale", None)
        table_cell_info.unit = table_info.get("units", None)
        table_cell_info.context_keywords = input_table.kwargs["keywords"]
        keyword_set = set()
        keyword_set.add(table_name)
        for table_cell in table_cell_info.cell_dict.values():
            table_cell: TableCell = table_cell
            keyword_set.update(list(table_cell.row_keywords.keys()))
        keywords_and_colloquial_expression = self._extract_keyword_from_table_header(
            keyword_set=keyword_set, table_name=table_name
        )
        for k, v in keywords_and_colloquial_expression.items():
            if k == table_name:
                table_cell_info.table_name_colloquial = v
                continue
            for table_cell in table_cell_info.cell_dict.values():
                if k in table_cell.row_keywords:
                    table_cell.row_keywords[k] = v
        return self.get_subgraph(
            input_table,
            table_df,
            table_cell_info,
        )

    def _extract_keyword_from_table_header(self, keyword_set: Set, table_name: str):
        context = table_name
        keyword_list = list(keyword_set)
        keyword_list.sort()
        input_dict = {"key_list": keyword_list, "context": context}
        keywords_and_colloquial_expression = self.llm.invoke(
            {
                "input": json.dumps(input_dict, ensure_ascii=False, sort_keys=True),
            },
            self.table_keywords_prompt,
            with_json_parse=True,
            with_except=True,
        )
        return keywords_and_colloquial_expression

    def _extract_simple_table(self, input_table: Chunk):
        rst = []
        if "table_info" in input_table.kwargs:
            table_info = input_table.kwargs.pop("table_info")
        else:
            table_info = {}
        if "header" in table_info and "index_col" in table_info:
            header = table_info["header"]
            index_col = table_info["index_col"]
            self._std_table(input_table=input_table, header=header, index_col=index_col)
        # 调用ner进行实体识别
        table_chunks = self.split_table(input_table, 500)
        for c in table_chunks:
            subgraph_lsit = self.schema_free_extractor.invoke(input=c)
            rst.extend(subgraph_lsit)
        return rst

    def _extract_other_table(self, input_table: Chunk):
        return self._extract_simple_table(input_table=input_table)

    def _std_table(self, input_table: Chunk, header: List, index_col: List):
        """
        按照表格新识别的表头，生成markdown文本
        """
        if "html" in input_table.kwargs:
            html = input_table.kwargs["html"]
            try:
                if len(header) <= 0:
                    header = None
                if not index_col or len(index_col) <= 0:
                    index_col = None
                table_df = pd.read_html(
                    StringIO(html),
                    header=header,
                    index_col=index_col,
                )[0]
            except IndexError:
                logging.exception("read html errro")
            table_df = table_df.fillna("")
            table_df = table_df.astype(str)
            input_table.content = table_df.to_markdown()
            del input_table.kwargs["html"]
            input_table.kwargs["content_type"] = "markdown"
            input_table.kwargs["csv"] = table_df.to_csv()
            return table_df, header, index_col

    def _std_table2(self, input_table: Chunk):
        """
        转换表格
        使用大模型转换
        """
        input_str = input_table.content
        # skip table reformat
        new_markdown = self.llm.invoke(
            {"input": input_str}, self.table_reformat, with_except=True
        )
        # new_markdown = input_str
        # new_markdown = new_markdown.replace("&nbsp;&nbsp;", "-")

        html_content = markdown.markdown(
            new_markdown, extensions=["markdown.extensions.tables"]
        )
        try:
            table_df = pd.read_html(StringIO(html_content), header=[0], index_col=[0])[
                0
            ]
        except ValueError:
            # 可能是表头数量没对齐，再尝试一次
            new_markdown = self._fix_llm_markdown(llm_markdown=new_markdown)
            html_content = markdown.markdown(
                new_markdown, extensions=["markdown.extensions.tables"]
            )
            table_df = pd.read_html(StringIO(html_content), header=[0], index_col=[0])[
                0
            ]

        table_df = table_df.fillna("")
        table_df = table_df.astype(str)
        input_table.content = table_df.to_markdown()
        input_table.kwargs.pop("html", None)
        input_table.kwargs["content_type"] = "markdown"
        input_table.kwargs["csv"] = table_df.to_csv()
        return table_df, [0], [0]

    def _fix_llm_markdown(self, llm_markdown: str):
        # 将输入的表格按行分割
        lines = llm_markdown.strip().split("\n")

        # 获取表头
        header = lines[0].strip()

        # 获取分隔行并根据 '|' 分割
        separator = lines[1].strip().split("|")

        # 确保 header 和每行都有前后的 '|'
        if not header.startswith("|"):
            header = "|" + header
        if not header.endswith("|"):
            header = header + "|"

        # 获取列数
        num_columns = header.count("|") - 1

        # 修复分隔行
        fixed_separator = "|" + "|".join(["---"] * num_columns) + "|"

        # 创建修复后的表格
        fixed_lines = [header, fixed_separator] + lines[2:]

        # 返回修复后的表格
        return "\n".join(fixed_lines)

    def get_subgraph(
        self, input_table: Chunk, table_df: pd.DataFrame, table_cell_info: TableInfo
    ):
        nodes = []
        edges = []
        import pickle

        # with open("table_data.pkl", "wb") as f:
        #     pickle.dump((input_table, table_df, table_cell_info), f)
        # print("write to table_data.pkl")

        table_id = input_table.id

        # keywords node
        keyword_str_set = set(table_cell_info.context_keywords)
        keywords = []

        for k, cell in table_cell_info.cell_dict.items():
            keyword_str_set = keyword_str_set.union(set(cell.row_keywords.keys()))

        for keyword in keyword_str_set:
            node = Node(
                _id=f"{table_id}-keyword{keyword}",
                name=keyword,
                label="TableKeyWord",
                properties={},
            )
            keywords.append(node.id)
            nodes.append(node)
        # Table node
        table_desc = input_table.kwargs["context"]
        table_name = input_table.kwargs["table_name"]
        table_node = Node(
            _id=table_id,
            name=table_name,
            label="Table",
            properties={
                "raw_name": table_name,
                "content": input_table.content,
                "csv": table_df.to_csv(),
                "desc": table_desc,
            },
        )
        nodes.append(table_node)

        # TableRow nodes
        idx = 0
        rows = {}
        levels = []
        for row_name, row_value in table_df.iterrows():
            node = Node(
                _id=f"{table_id}-row-{idx}",
                name=f"{table_name}-{row_name.lstrip('-').strip()}",
                label="TableRow",
                properties={
                    "raw_name": row_name.lstrip("-").strip(),
                    "content": row_value.to_csv(),
                    "desc": table_desc,
                },
            )
            rows[idx] = (row_name.lstrip("-").strip(), node.id)
            row_level = 0
            for c in row_name:
                if c != "-":
                    break
                else:
                    row_level += 1
            levels.append((idx, row_level, node.id))
            idx += 1
            nodes.append(node)
        # TableCol nodes
        idx = 0
        cols = {}
        for col_name, col_value in table_df.items():
            node = Node(
                _id=f"{table_id}-col-{idx}",
                name=f"{table_name}-{col_name}",
                label="TableColumn",
                properties={
                    "raw_name": col_name,
                    "content": col_value.to_csv(),
                    "desc": table_desc,
                },
            )
            cols[idx] = (col_name, node.id)
            idx += 1
            nodes.append(node)

        # Table cells
        cells = {}
        for k, cell in table_cell_info.cell_dict.items():
            row_num, col_num = k.split("-")
            row_num = int(row_num)
            col_num = int(col_num)
            row_name, _ = rows[row_num]
            col_name, _ = cols[col_num]
            table_cell: TableCell = cell
            cell_id = f"{table_id}-{k}"
            # cell node
            node = Node(
                _id=cell_id,
                name=f"{table_name}-{row_name}-{col_name}",
                label="TableCell",
                properties={
                    "raw_name": f"{row_name}-{col_name}",
                    "row_name": row_name,
                    "col_name": col_name,
                    "desc": table_cell.desc,
                    "value": table_cell.value,
                    "scale": table_cell_info.sacle,
                    "unit": table_cell_info.unit,
                },
            )
            cells[(row_num, col_num)] = node.id
            nodes.append(node)
        node_map = {}
        for node in nodes:
            node_map[node.id] = node
        # table <-> row
        for k, v in rows.items():
            row_name, row_id = v
            edge = Edge(
                _id=f"table-{table_id}-col-{row_id}",
                from_node=node_map[table_id],
                to_node=node_map[row_id],
                label="containRow",
                properties={},
            )

            edges.append(edge)

            edge = Edge(
                _id=f"row-{row_id}-table-{table_id}",
                from_node=node_map[row_id],
                to_node=node_map[table_id],
                label="partOf",
                properties={},
            )
            edges.append(edge)

        # table <-> col
        for k, v in cols.items():
            col_name, col_id = v
            edge = Edge(
                _id=f"table-{table_id}-col-{col_id}",
                from_node=node_map[table_id],
                to_node=node_map[col_id],
                label="containColumn",
                properties={},
            )
            edges.append(edge)

            edge = Edge(
                _id=f"col-{col_id}-table-{table_id}",
                from_node=node_map[col_id],
                to_node=node_map[table_id],
                label="partOf",
                properties={},
            )
            edges.append(edge)

        # table/row/col <-> table cell
        for cell_loc, cell_id in cells.items():
            row_num, col_num = cell_loc
            row_id = rows[row_num][1]
            col_id = cols[col_num][1]
            edge = Edge(
                _id=f"row-{row_id}-contain-cell-{cell_id}",
                from_node=node_map[row_id],
                to_node=node_map[cell_id],
                label="containCell",
                properties={},
            )
            edges.append(edge)
            edge = Edge(
                _id=f"cell-{cell_id}-part-of-row-{row_id}",
                from_node=node_map[cell_id],
                to_node=node_map[row_id],
                label="partOfTableRow",
                properties={},
            )
            edges.append(edge)

            edge = Edge(
                _id=f"col-{col_id}-contain_cell-{cell_id}",
                from_node=node_map[col_id],
                to_node=node_map[cell_id],
                label="containCell",
                properties={},
            )
            edges.append(edge)
            edge = Edge(
                _id=f"cell-{cell_id}-part-of-col-{col_id}",
                from_node=node_map[cell_id],
                to_node=node_map[col_id],
                label="partOfTableColumn",
                properties={},
            )
            edges.append(edge)

            edge = Edge(
                _id=f"cell-{cell_id}-part-of-table-{table_id}",
                from_node=node_map[cell_id],
                to_node=node_map[col_id],
                label="partOfTable",
                properties={},
            )

        # row subitem
        for i in range(len(levels)):
            # row_num = levels[i][0]
            # row_info = levels[i][1]
            row_num, row_level, row_id = levels[i]
            if row_level != 0:
                for j in range(i - 1, -1, -1):
                    tmp_row_num, tmp_row_level, tmp_row_id = levels[j]
                    # tmp_row_info = levels[j][1]

                    # tmp_row_level, tmp_row_id = tmp_row_info
                    if tmp_row_level < row_level:
                        edge = Edge(
                            _id=f"row-{row_id}-sub-item-{tmp_row_id}",
                            from_node=node_map[tmp_row_id],
                            to_node=node_map[row_id],
                            label="subitem",
                            properties={},
                        )
                        edges.append(edge)
                        break

        # table ->keyword
        for keyword_id in keywords:
            edge = Edge(
                _id=f"keyword-{keyword_id}-table-{table_id}",
                from_node=node_map[keyword_id],
                to_node=node_map[table_id],
                label="keyword",
                properties={},
            )
            edges.append(edge)

        subgraph = SubGraph(nodes=nodes, edges=edges)
        print("*" * 80)
        print(
            f"done process {table_df.shape} table to subgraph with {len(nodes)} nodes and {len(edges)} edges"
        )
        print(f"node stat: {self.stat(nodes)}")
        print(f"edge stat: {self.stat(edges)}")
        return [subgraph]

    def stat(self, items):
        out = {}
        for item in items:
            label = item.label
            if label in out:
                out[label] += 1
            else:
                out[label] = 1
        return out

    def _get_colloquial_nodes_and_edges(
        self,
        colloquial_list: List,
        table_name: str,
        all_keywords_dict: Dict,
        splited_keyword_node: Node,
    ):
        nodes = []
        edges = []
        for ck in colloquial_list:
            colloquial_keyword: str = ck
            c_keyword_id = f"{table_name}_{colloquial_keyword}"
            if c_keyword_id in all_keywords_dict:
                continue
            c_keyword_node = Node(
                _id=c_keyword_id,
                name=colloquial_keyword,
                label="MetricConstraint",
                properties={"type": "colloquial"},
            )
            all_keywords_dict[c_keyword_id] = c_keyword_node
            nodes.append(c_keyword_node)

            # keyword to row_keyword
            edge = Edge(
                _id="k2rk2c_" + c_keyword_id,
                from_node=c_keyword_node,
                to_node=splited_keyword_node,
                label="colloquial",
                properties={},
            )
            edges.append(edge)
        return nodes, edges

    def _generate_table_cell_info(
        self,
        data: pd.DataFrame,
        header,
        table_name,
        cell_value_desc,
    ):
        table_info = TableInfo(table_name=table_name)
        sub_item_dict = {}

        def format_value(value):
            value = value.replace("(", "").replace(")", "").replace(",", "")
            if value.endswith("%"):
                # 去除百分号并转换为浮点数
                return float(value.rstrip("%")) / 100
            elif value.isdigit():
                return float(value)
            else:
                return None

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data.iloc[i, j]
                # value = format_value(value)
                # if not value:
                #    continue
                x_index = i + len(header)
                y_index = j + 1
                cell_id = f"{x_index-1}-{y_index-1}"
                row_keywords = {}
                describe = ""
                now_index_str = None
                if pd.isnull(data.index[i]):
                    describe += "total"
                    row_keywords["total"] = {}
                else:
                    now_index_str = f"{data.index[i]}"
                    describe += now_index_str
                    row_keywords[now_index_str] = {}
                temp_i = i - 1
                while temp_i >= 0:
                    if (data.iloc[temp_i] == "").all():
                        parent_str = f"{data.index[temp_i]}"
                        parent_str = parent_str.strip(":").strip("：")
                        describe += f" in {parent_str}"
                        row_keywords[parent_str] = {}
                        if now_index_str is not None:
                            sub_item_set = sub_item_dict.get(parent_str, set())
                            sub_item_set.add(now_index_str)
                            sub_item_dict[parent_str] = sub_item_set
                        break
                    temp_i -= 1

                describe += " of"
                if len(header) == 0:
                    pass
                elif len(header) == 1:
                    header_str = self._handle_unnamed_single_topheader(data.columns, j)
                    describe += f" {header_str}"
                    row_keywords[header_str] = {}
                else:
                    header_str = self._handle_unnamed_multi_topheader(data.columns, j)
                    describe += f" {header_str}"
                    row_keywords[header_str] = {}
                    prev = self._handle_unnamed_multi_topheader(data.columns, j)
                    for temp_j in header[1:]:
                        if (
                            data.columns[j][temp_j].startswith("Unnamed")
                            or data.columns[j][temp_j] == ""
                        ):
                            continue
                        if data.columns[j][temp_j] == prev:
                            continue
                        describe += f" {data.columns[j][temp_j]}"
                        row_keywords[f"{data.columns[j][temp_j]}"] = {}
                        prev = data.columns[j][temp_j]
                describe += f" is {data.iloc[i, j]}{cell_value_desc}"
                describe = f"[{table_name}]cell[{cell_id}] shows " + describe
                table_cell = TableCell(desc=describe, row_keywords=row_keywords)
                table_cell.value = data.iloc[i, j]
                table_info.cell_dict[cell_id] = table_cell
        table_info.sub_item_dict = sub_item_dict
        return table_info

    def _handle_unnamed_single_topheader(self, columns, j):
        tmp = j
        while tmp < len(columns) and (
            columns[tmp].startswith("Unnamed") or columns[tmp] == ""
        ):
            tmp += 1
        if tmp < len(columns):
            return columns[tmp]

        tmp = j
        while tmp >= 0 and (columns[tmp].startswith("Unnamed") or columns[tmp] == ""):
            tmp -= 1
        if tmp < 0:
            return f"data {j}"
        else:
            return columns[tmp]

    def _handle_unnamed_multi_topheader(self, columns, j):
        tmp = j
        while tmp < len(columns) and (
            columns[tmp][0].startswith("Unnamed") or columns[tmp][0] == ""
        ):
            tmp += 1
        if tmp < len(columns):
            return columns[tmp][0]

        tmp = j
        while tmp >= 0 and (
            columns[tmp][0].startswith("Unnamed") or columns[tmp][0] == ""
        ):
            tmp -= 1
        if tmp < 0:
            return f"data {j}"
        else:
            return columns[tmp][0]

    def split_table(self, org_chunk: Chunk, chunk_size: int = 2000, sep: str = "\n"):
        """
        Internal method to split a markdown format table into smaller markdown tables.

        Args:
            org_chunk (Chunk): The original chunk containing the table data.
            chunk_size (int): The maximum size of each smaller chunk. Defaults to 2000.
            sep (str): The separator used to join the table rows. Defaults to "\n".

        Returns:
            List[Chunk]: A list of smaller chunks resulting from the split operation.
        """
        output = []
        content = org_chunk.content
        table_start = content.find("|")
        table_end = content.rfind("|") + 1
        if table_start is None or table_end is None or table_start == table_end:
            return None
        prefix = content[0:table_start].strip("\n ")
        table_rows = content[table_start:table_end].split("\n")
        table_header = table_rows[0]
        table_header_segmentation = table_rows[1]
        suffix = content[table_end:].strip("\n ")

        splitted = []
        cur = [prefix, table_header, table_header_segmentation]
        cur_len = len(prefix)
        for idx, row in enumerate(table_rows[2:]):
            if cur_len > chunk_size:
                cur.append(suffix)
                splitted.append(cur)
                cur_len = 0
                cur = [prefix, table_header, table_header_segmentation]
            cur.append(row)
            cur_len += len(row)

        cur.append(content[table_end:])
        if len(cur) > 0:
            splitted.append(cur)

        output = []
        for idx, sentences in enumerate(splitted):
            chunk = Chunk(
                id=f"{org_chunk.id}#{chunk_size}#table#{idx}#LEN",
                name=f"{org_chunk.name}#{idx}",
                content=sep.join(sentences),
                type=org_chunk.type,
                **org_chunk.kwargs,
            )
            output.append(chunk)
        return output
