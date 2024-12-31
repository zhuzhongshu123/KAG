# KAG Example: CSQA

The [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain/tree/main)
``cs.jsonl`` dataset contains 10 documents in Computer Science and
100 questions with their answers about those documents.

Here we demonstrate how to build a knowledge graph for those documents,
generate answers to those questions with KAG and compare KAG generated
answers with those from other RAG systems.

## Steps to reproduce

1. Follow the Quick Start guide of KAG to install OpenSPG server and KAG.

2. (Optional) Download [UltraDomain](https://huggingface.co/datasets/TommyChien/UltraDomain/tree/main)
   ``cs.jsonl`` and execute ``generate_data.py`` to generate data files in
   ``./builder/data`` and ``./solver/data``. Since the generated files
   was committed, this step is optional.

3. Update ``llm`` and ``vectorizer_model`` configuration in ``kag_config.yaml``
   properly. ``splitter`` and ``num_threads_per_chain`` may also be updated
   to match with other systems.

4. Restore the KAG project.

   ```bash
   knext project restore --host_addr http://127.0.0.1:8887 --proj_path .
   ```

5. Commit the schema.

   ```bash
   knext schema commit
   ```

6. Execute ``indexer.py`` in the ``builder`` directory to build the knowledge graph.

7. Execute ``eval.py`` in the ``solver`` directory to generate the answers.

   The results are saved to ``./solver/data/csqa_kag_answers.json``.

8. (Optional) Follow the LightRAG [Reproduce](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#reproduce)
   steps to generate answers to the questions and save the results to
   ``./solver/data/csqa_lightrag_answers.json``. Since a copy was committed,
   this step is optional.

9. Update LLM configurations in ``summarization_metrics.py`` and ``factual_correctness.py``
   and execute them to get the metrics.