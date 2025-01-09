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
import copy
import logging
from typing import Dict, Type, List

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
        #return self.schema_free_extractor._invoke(input, **kwargs)
        return []

    def _invoke_table(self, input: Input, **kwargs) -> List[Output]:
        print("one_table")
        return []
