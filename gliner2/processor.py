import copy
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import torch
from transformers import AutoTokenizer


class TokenSplitterBase():
    def __init__(self):
        pass

    def __call__(self, text) -> (str, int, int):
        pass


class WhitespaceTokenSplitter(TokenSplitterBase):
    """
    One‑pass tokenizer that treats…
      • URLs   → single token
      • Emails → single token
      • @handles → single token
      • Words with optional -/_ inside → single token
      • Any other non‑space char → single token
    Yields (token, start, end) like nltk.tokenize.RegexTokenizer.
    """

    __slots__ = ()                     # tiny memory footprint

    _PATTERN = re.compile(             # pre‑compiled once at import time
        r"""
        (?:https?://[^\s]+|            # URL with scheme
           www\.[^\s]+)                # or bare www.
        |                              # ───────────────────
        [a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}   # e‑mail
        |                              # ───────────────────
        @[a-z0-9_]+                    # social handle (@foo_bar)
        |                              # ───────────────────
        \w+(?:[-_]\w+)*                # classic word (foo-bar_baz)
        |                              # ───────────────────
        \S                             # fallback single char
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(
        self, text: str, lower: bool = True
    ) -> Iterator[Tuple[str, int, int]]:
        if lower:
            text = text.lower()
        for m in self._PATTERN.finditer(text):
            yield m.group(), m.start(), m.end()


class SamplingConfig:
    def __init__(
            self,
            # JSON Structures
            remove_json_structure_prob: float = 0.2,
            shuffle_json_fields: bool = True,
            remove_json_field_prob: float = 0.2,
            # Entities
            remove_entities_prob: float = 0.,
            shuffle_entities: bool = False,
            remove_entity_prob: float = 0.,
            synthetic_entity_label_prob: float = 0.2,
            # Relations
            remove_relations_prob: float = 0.2,
            swap_head_tail_prob: float = 0.2,
            # Classifications
            remove_classification_prob: float = 0,
            shuffle_classification_labels: bool = True,
            remove_classification_label_prob: float = 0.5,
            synthetic_label_prob: float = 0.5,
            include_true_label_prob: float = 0.5,  # Probability to include the true label in classifications
            max_num_labels: int = 1000  # Limit to avoid OOM in case of too many labels
    ):
        """
        Configuration for stochastic sampling during training.

        Args:
            remove_json_structure_prob (float): Probability to remove an entire JSON structure.
            shuffle_json_fields (bool): Whether to shuffle the order of fields in JSON structures.
            remove_json_field_prob (float): Probability to remove a field within a JSON structure.
            remove_entities_prob (float): Probability to remove the entire entities structure.
            synthetic_entity_label_prob (float): Probability to use synthetic entity labels.
            shuffle_entities (bool): Whether to shuffle the order of entities.
            remove_entity_prob (float): Probability to remove an entity within the structure.
            remove_relations_prob (float): Probability to remove the entire relation structure.
            swap_head_tail_prob (float): Probability to swap head and tail order in relations.
            remove_classification_prob (float): Probability to remove the entire classification structure.
            shuffle_classification_labels (bool): Whether to shuffle the order of classification labels.
            remove_classification_label_prob (float): Probability to remove a classification label.
            synthetic_label_prob (float): Probability to use synthetic labels for classifications.
            include_true_label_prob (float): Probability to include the true label in classifications.
            max_num_labels (int): Limit on the number of classification labels to avoid OOM.
        """
        # JSON Structures
        self.remove_json_structure_prob = remove_json_structure_prob
        self.shuffle_json_fields = shuffle_json_fields
        self.remove_json_field_prob = remove_json_field_prob

        # Entities
        self.remove_entities_prob = remove_entities_prob
        self.shuffle_entities = shuffle_entities
        self.remove_entity_prob = remove_entity_prob
        self.synthetic_entity_label_prob = synthetic_entity_label_prob

        # Relations
        self.remove_relations_prob = remove_relations_prob
        self.swap_head_tail_prob = swap_head_tail_prob

        # Classifications
        self.remove_classification_prob = remove_classification_prob
        self.shuffle_classification_labels = shuffle_classification_labels
        self.remove_classification_label_prob = remove_classification_label_prob
        self.synthetic_label_prob = synthetic_label_prob
        self.include_true_label_prob = include_true_label_prob
        self.max_num_labels = max_num_labels


class SchemaTransformer:
    def __init__(self, model_name: str = None, tokenizer=None, sampling_config: SamplingConfig = None):
        """
        Initialize the SchemaTransformer with a pretrained tokenizer and sampling configuration.
        If sampling_config is None, a default SamplingConfig is used.

        Args:
            model_name (str): Name or path of the model to load the tokenizer.
            tokenizer: An already loaded tokenizer (if provided, model_name is ignored).
            sampling_config (SamplingConfig): Custom sampling configuration for data augmentation.
        """
        if model_name is None and tokenizer is None:
            raise ValueError("Either model_name or tokenizer must be provided.")

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.word_splitter = WhitespaceTokenSplitter()

        # Special tokens used for schema processing
        self.sep_struct = "[SEP_STRUCT]"
        self.sep_text = "[SEP_TEXT]"
        self.p_token = "[P]"
        self.c_token = "[C]"
        self.e_token = "[E]"
        self.r_token = "[R]"
        self.l_token = "[L]"

        self.example_token = "[EXAMPLE]"
        self.output_example_token = "[OUTPUT]"
        self.description_token = "[DESCRIPTION]"

        # Add special tokens to the tokenizer if they're not already present
        special_tokens = {
            "additional_special_tokens": [
                self.sep_struct,
                self.sep_text,
                self.p_token,
                self.c_token,
                self.e_token,
                self.r_token,
                self.l_token,
                self.example_token,
                self.output_example_token,
                self.description_token
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Set training mode flag
        self.is_training = False

        # Use the provided sampling configuration if available; otherwise, use defaults.
        self.sampling_config = sampling_config if sampling_config is not None else SamplingConfig()

    def change_mode(self, is_training: bool):
        self.is_training = is_training

    def build_classification_prefix(self, schema: Dict[str, Any]) -> List[str]:
        """
        Returns a flat list of tokens encoding in-JSON classification fields as a prefix.
        Uses minimal token splitting (e.g., 'hotel reservation' is a single token).
        During training, both the order of fields and choices are shuffled for augmentation.
        Example:
        ['(', 'hotel reservation:', 'is_resort', '(', 'yes', '|', 'no', ')', ',', 'hotel type', '(', ... ')', ')']
        """
        prefix_tokens = []

        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                # Extract candidate classification fields
                classification_fields = [
                    (fname, fval) for fname, fval in fields.items()
                    if isinstance(fval, dict) and "value" in fval and "choices" in fval
                ]

                if self.is_training:
                    random.shuffle(classification_fields)

                inner_tokens = []
                for fname, fval in classification_fields:
                    choices = fval["choices"].copy()
                    if self.is_training:
                        random.shuffle(choices)

                    choice_tokens = []
                    for i, choice in enumerate(choices):
                        choice_tokens.append(choice)
                        if i < len(choices) - 1:
                            choice_tokens.append('|')

                    inner_tokens.extend([fname, '('] + choice_tokens + [')', ','])

                if inner_tokens:
                    inner_tokens = inner_tokens[:-1]  # remove trailing comma
                    prefix_tokens.extend(['(', f"{parent}:", *inner_tokens, ')'])

        return prefix_tokens

    ## @staticmethod
    def tokenize_text(self, text: str, lower: bool = True) -> List[str]:
        """
        Tokenize the input text into a list of tokens (words/punctuation).
        """
        # if lower:
        #     text = text.lower()
        # return re.findall(r'\w+(?:[-_]\w+)*|\S', text)
        # use self.word_splitter to tokenize the text
        # tokens = []
        # for token, start, end in WhitespaceTokenSplitter()(text, lower=lower):
        #     tokens.append(token)
        return [tok for tok, _, _ in self.word_splitter(text, lower=lower)]

    def transform_schema(
            self,
            parent: str,
            fields: List[str],
            child_prefix: str = "[C]",
            prompt: Optional[str] = None,
            examples: Optional[List[Tuple[str, str]]] = None,
            label_descriptions: Optional[Dict[str, str]] = None,
            example_mode: str = "both"  # Options: "descriptions", "few_shot", "both", "none"
    ) -> List[str]:
        """
        Transforms the provided schema into a linear token structure with optional prompt,
        few-shot examples, and label descriptions.

        Args:
            parent: The name of the task (e.g., "hotel_booking", "event", "entities").
            fields: A list of field names for the task.
            child_prefix: The prefix for each field (e.g., "[C]", "[E]", "[R]", "[L]").
            prompt: An optional instruction or prompt to concatenate with the parent.
            examples: Optional few-shot examples as a list of (input, output) tuples.
            label_descriptions: Optional dict mapping label names to their descriptions.
            example_mode: Controls inclusion ("descriptions", "few_shot", "both", "none").
        """
        # Build the main prompt string
        prompt_str = parent
        if prompt:
            prompt_str = f"{parent}: {prompt}"

        # Append label descriptions if applicable
        if example_mode in ["descriptions", "both"] and label_descriptions:
            # Filter and shuffle descriptions corresponding to fields
            descriptions = [(label, desc) for label, desc in label_descriptions.items() if label in fields]
            if self.is_training:
                random.shuffle(descriptions)
            for label, desc in descriptions:
                prompt_str += f" {self.description_token} {label}: {desc}"

        # Append few-shot examples if applicable
        if example_mode in ["few_shot", "both"] and examples:
            if self.is_training:
                random.shuffle(examples)
            for input_text, output_label in examples:
                if output_label in fields:
                    if isinstance(output_label, str):
                        prompt_str += f" {self.example_token} {input_text} {self.output_example_token} {output_label}"
                    else:
                        prompt_str += f" {self.example_token} {input_text} {self.output_example_token} {', '.join(output_label)}"

        # Construct the schema tokens
        schema_tokens = ["(", "[P]", prompt_str, "("]
        for field in fields:
            schema_tokens.extend([child_prefix, field])
        schema_tokens.extend([")", ")"])

        return schema_tokens

    def infer_from_json(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infers schema tokens and structure labels from a JSON schema with stochastic sampling
        applied during training. Sampling modifications include:

          - json_structures:
              * Randomly remove an entire structure.
              * Shuffle the order of fields.
              * Randomly remove individual fields.
              * Ensure that for a given parent, if at least one occurrence is kept, all occurrences
                are kept and use the same field ordering.
          - entities:
              * Randomly remove the entire structure.
              * Shuffle the entities.
              * Randomly remove individual entities.
          - relations:
              * Randomly remove an entire structure.
              * With some probability, swap the order of 'head' and 'tail' (applied consistently).
          - classifications:
              * Randomly remove an entire structure.
              * Shuffle the order of labels.

        Sampling is applied only if in training mode and if a sampling configuration is set.

        Returns:
            A dictionary containing:
              - "schemas": list of linear token structures (one per task)
              - "structure_labels": list of structure labels (for non-classification tasks)
              - "task_types": list of task types corresponding to each schema.
        """

        inferred_schemas: List[List[str]] = []
        structure_labels: List[Any] = []
        task_types: List[str] = []

        # Use sampling configuration only in training mode.
        sampling = self.sampling_config if (hasattr(self, "sampling_config") and self.is_training) else None

        # --- Process JSON Structures (group by parent) ---
        if "json_structures" in schema:
            json_descriptions = schema.get("json_descriptions", {})

            groups: Dict[str, List[Dict[str, Any]]] = {}
            for item in schema["json_structures"]:
                for parent, fields in item.items():
                    groups.setdefault(parent, []).append(fields)

            for parent, occurrences in groups.items():
                if sampling and random.random() < sampling.remove_json_structure_prob:
                    continue

                all_fields = set()
                for occ in occurrences:
                    all_fields.update(occ.keys())
                common_fields = list(all_fields)

                if sampling and sampling.shuffle_json_fields:
                    random.shuffle(common_fields)

                chosen_fields = [f for f in common_fields if
                                 not (sampling and random.random() < sampling.remove_json_field_prob)]
                if not chosen_fields:
                    continue

                # example_mode_list
                example_mode_list = ["none", "descriptions"]  # if self.is_training else ["descriptions"]

                # Keep a copy of the real field names for spans
                real_field_order = chosen_fields.copy()
                real2syn: Dict[str, str] = {}
                # Synthetic labeling for fields
                if sampling and random.random() < sampling.synthetic_entity_label_prob:
                    example_mode_list.remove("none")
                    synthetic_fields = []
                    for i, real in enumerate(real_field_order, start=1):
                        syn = f"field {i}"
                        real2syn[real] = syn
                        synthetic_fields.append(syn)
                    chosen_fields = synthetic_fields
                    # Build synthetic descriptions with fallback
                    orig_descs = json_descriptions.get(parent, {})
                    synthetic_descs: Dict[str, str] = {}
                    for real in real_field_order:
                        syn = real2syn[real]
                        desc = orig_descs.get(real, real)
                        synthetic_descs[syn] = desc
                    json_descriptions[parent] = synthetic_descs

                # spans & labels
                spans: List[List[Any]] = []
                for occ in occurrences:
                    span = [occ.get(f, None) for f in chosen_fields]
                    spans.append(span)

                if not spans:
                    continue

                # dedup
                uniq: List[List[Any]] = []
                seen = set()
                for s in spans:
                    key = tuple(tuple(x) if isinstance(x, list) else x for x in s)
                    if key not in seen:
                        uniq.append(s)
                        seen.add(key)

                # --- after you compute `uniq` (the deduped list of spans) ---
                # if every cell in every span is empty ("" or None), treat as “no spans”
                if all(all((cell is None) or (cell == "") for cell in span) for span in uniq):
                    count = 0
                    uniq = []
                else:
                    count = len(uniq)

                structure_labels.append([count, uniq])

                # description logic
                label_descs = json_descriptions.get(parent, {})
                if self.is_training:
                    example_mode = random.choice(example_mode_list)
                else:
                    example_mode = "descriptions" if label_descs else "none"

                inferred_schemas.append(
                    self.transform_schema(
                        parent,
                        chosen_fields,
                        child_prefix=self.c_token,
                        label_descriptions=label_descs,
                        example_mode=example_mode,
                    )
                )
                task_types.append("json_structures")

        if "entities" in schema:
            if not (sampling and random.random() < sampling.remove_entities_prob):
                entity_fields = list(schema["entities"].keys())
                entity_descs = schema.get("entity_descriptions", {})
                example_mode_list = ["none", "descriptions"]  # if self.is_training else ["descriptions"]

                real2syn: Dict[str, str] = {}
                if sampling and random.random() < sampling.synthetic_entity_label_prob:
                    example_mode_list.remove("none")
                    synthetic = []
                    for i, real in enumerate(entity_fields, 1):
                        syn = f"entity {i}"
                        real2syn[real] = syn
                        synthetic.append(syn)
                    new_descs = {real2syn.get(k, k): v for k, v in entity_descs.items()}
                    entity_descs = new_descs
                    schema["entities"] = {real2syn.get(k, k): v for k, v in schema["entities"].items()}
                    entity_fields = synthetic

                if sampling and sampling.shuffle_entities:
                    random.shuffle(entity_fields)

                chosen_entities = [e for e in entity_fields if
                                   not (sampling and random.random() < sampling.remove_entity_prob)]
                if chosen_entities:
                    span = [schema["entities"][e] for e in chosen_entities]
                    structure_labels.append([1, [span]])

                    example_mode = random.choice(example_mode_list) if self.is_training else (
                        "descriptions" if entity_descs else "none"
                    )

                    inferred_schemas.append(
                        self.transform_schema(
                            "entities",
                            chosen_entities,
                            child_prefix=self.e_token,
                            label_descriptions=entity_descs,
                            example_mode=example_mode,
                        )
                    )
                    task_types.append("entities")

        # --- Process Relations (group by relation name) ---
        if "relations" in schema:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for item in schema["relations"]:
                if sampling is not None and random.random() < sampling.remove_relations_prob:
                    continue
                for parent, fields in item.items():
                    groups.setdefault(parent, []).append(fields)
            for parent, occurrences in groups.items():
                # Decide on a common field set; use keys from first occurrence.
                field_names = list(occurrences[0].keys())
                # Optionally swap 'head' and 'tail' consistently.
                if sampling is not None and "head" in field_names and "tail" in field_names:
                    if random.random() < sampling.swap_head_tail_prob:
                        idx_head = field_names.index("head")
                        idx_tail = field_names.index("tail")
                        field_names[idx_head], field_names[idx_tail] = field_names[idx_tail], field_names[idx_head]
                spans = []
                for occ in occurrences:
                    if all(f in occ for f in field_names):
                        span = [occ[f] for f in field_names]
                        spans.append(span)
                if not spans:
                    continue
                seen = set()
                unique_spans = []
                for span in spans:
                    t = tuple([tuple(s) if isinstance(s, list) else s for s in span])
                    if t not in seen:
                        seen.add(t)
                        unique_spans.append(span)
                count = len(unique_spans)
                structure_labels.append([count, unique_spans])
                inferred_schemas.append(self.transform_schema(parent, field_names, child_prefix="[R]"))
                task_types.append("relations")

        # --- Process Classifications (do not produce structure labels) ---
        if "classifications" in schema:
            for l, item in enumerate(schema["classifications"]):
                # Optionally drop entire classification
                if sampling is not None and random.random() < sampling.remove_classification_prob:
                    continue

                # --- inside infer_from_json(), in the classifications block ---
                labels = item["labels"].copy()
                examples = item.get("examples", [])
                label_descs = item.get("label_descriptions", {}) or {}

                # Decide whether to switch to synthetic names
                real2syn = {}
                if not self.is_training:
                    example_mode_list = ["both"]  # include descriptions and few-shot examples
                else:
                    example_mode_list = ["few_shot", "descriptions", "both", "none"]  # all modes during training

                if sampling is not None and random.random() < sampling.synthetic_label_prob:
                    # remove "none" from the example mode since it doesn't make sense to have synthetic labels
                    example_mode_list.remove("none")

                    # Build mapping: real → synthetic
                    synthetic_labels = []
                    for i, real in enumerate(labels, start=1):
                        syn = f"label {i}"
                        real2syn[real] = syn
                        synthetic_labels.append(syn)

                    # Replace the label list
                    labels = synthetic_labels

                    # Remap descriptions: if no real desc, use the real name as fallback
                    new_descs = {}
                    for real, syn in real2syn.items():
                        desc = label_descs.get(real, real)
                        new_descs[syn] = desc
                    label_descs = new_descs

                    # Remap any few‐shot examples’ output labels
                    new_examples = []
                    for inp, out in examples:
                        if out in real2syn:
                            new_examples.append((inp, real2syn[out]))
                        else:
                            new_examples.append((inp, out))
                    examples = new_examples

                if len(example_mode_list) == 1:
                    example_mode = example_mode_list[0]
                else:
                    example_mode = random.choice(example_mode_list)

                # Sample an actual drop probability up to the max
                if sampling is not None and hasattr(sampling, "remove_classification_label_prob"):
                    # 1) First, draw a value u ∼ Beta(0.5, 0.5).  This concentrates mass near 0 and near 1.
                    #    Then scale it by the max probability (remove_classification_label_prob).
                    alpha, beta = 1, 1
                    u = random.betavariate(alpha, beta)
                    drop_fraction = u * sampling.remove_classification_label_prob

                    # 2) Compute how many labels to remove based on that fraction:
                    num_remove = int(len(labels) * drop_fraction)

                    if num_remove > 0:
                        labels = random.sample(labels, len(labels) - num_remove)

                    # avoid OOM in case of too many labels
                    if example_mode in ["few_shot", "both", "descriptions"]:
                        if len(labels) > sampling.max_num_labels // 2:
                            labels = labels[:sampling.max_num_labels // 2]
                    else:
                        if len(labels) > sampling.max_num_labels:
                            labels = labels[:sampling.max_num_labels]

                    # 3) Still preserve the true label with the same logic as before:
                    if random.random() < sampling.include_true_label_prob:
                        true_label = schema["classifications"][l].get("true_label", [])
                        # if true_label not in labels:
                        if isinstance(true_label, list):
                            for true_label_item in true_label:
                                if true_label_item not in labels:
                                    labels.append(true_label_item)
                        else:
                            if true_label not in labels:
                                labels.append(true_label)

                if sampling is not None and sampling.shuffle_classification_labels:
                    random.shuffle(labels)

                inferred_schemas.append(
                    self.transform_schema(
                        item["task"],
                        labels,
                        child_prefix="[L]",
                        prompt=item.get("prompt"),
                        examples=examples,
                        label_descriptions=label_descs,
                        example_mode=example_mode
                    )
                )

                task_types.append("classifications")

                # Reorganize the classification labels in the original schema
                schema["classifications"][l]["labels"] = labels
                true_label = schema["classifications"][l]["true_label"].copy()
                schema["classifications"][l]["true_label"] = [real2syn.get(i, i) for i in true_label]
                structure_labels.append([])

        # --- Shuffle the order of tasks consistently ---
        if sampling is not None:
            order_idx = list(range(len(task_types)))
            random.shuffle(order_idx)
            inferred_schemas = [inferred_schemas[i] for i in order_idx]
            structure_labels = [structure_labels[i] for i in order_idx]
            task_types = [task_types[i] for i in order_idx]

        return {"schemas": inferred_schemas, "structure_labels": structure_labels, "task_types": task_types,
                "new_schema": schema}

    def transform_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single record into a format suitable for model training/inference.

        Parameters
        ----------
        record : Dict[str, Any]
            Dictionary containing:
                text : str
                    The input text to process
                schema : Dict
                    Schema information including classifications and structure definitions

        Returns
        -------
        Dict[str, Any]
            Processed record containing:
                schema_tokens_list : List[List[str]]
                    Tokenized schema structures
                text_tokens : List[str]
                    Tokenized input text
                outputs : List[Union[List, List[int]]]
                    Task outputs - either span positions or classification labels
                task_types : List[str]
                    Types of tasks corresponding to each output
        """
        record_ = copy.deepcopy(record)
        text, schema = record_["text"], record_["schema"]

        # --- prepend classification choices prefix (Option 1) ---
        prefix = self.build_classification_prefix(schema)
        if prefix:
            def wrap(val):
                """Prepend [selection] to a single value or each item in a list."""
                if isinstance(val, list):
                    return [f"[selection]{v}" for v in val]
                return f"[selection]{val}"

            # 1) Identify all “parent.field” names that are classifications
            class_keys = {
                f"{parent}.{fname}"
                for struct in schema.get("json_structures", [])
                for parent, fields in struct.items()
                for fname, fval in fields.items()
                if isinstance(fval, dict) and {"value", "choices"} <= fval.keys()
            }

            # 2) Flatten them all in one pass
            for struct in schema.get("json_structures", []):
                for parent, fields in struct.items():
                    for fname in list(fields):
                        key = f"{parent}.{fname}"
                        if key not in class_keys:
                            continue

                        fval = fields[fname]
                        # if it’s a dict, pull out .value; otherwise use the raw field
                        raw = fval["value"] if isinstance(fval, dict) else fval
                        fields[fname] = wrap(raw)

        # Tokenize the text
        text_tokens = []  # self.tokenize_text(text)
        start_token_idx_to_text_idx = []
        end_token_idx_to_text_idx = []
        for tkns, start, end in self.word_splitter(text, lower=True):
            text_tokens.append(tkns)
            start_token_idx_to_text_idx.append(start)
            end_token_idx_to_text_idx.append(end)

        if prefix:
            text_tokens = prefix + text_tokens

        len_prefix = len(prefix) if prefix else 0

        # Infer the schema tokens and structure labels
        processed = self.infer_from_json(schema)
        processed_schemas = processed["schemas"]
        structure_labels = processed.get("structure_labels", [])
        task_types = processed["task_types"]

        schema = processed["new_schema"]

        results = []

        def find_sublist(sub: List[str], lst: List[str]) -> List[Tuple[int, int]]:
            """
            Return **all** (start, end) index pairs where the sub-list ``sub`` occurs
            inside ``lst``.
            If the sub-list never appears, returns a single sentinel [(-1, -1)].

            Examples
            --------
            >>> find_sublist(['a', 'b'], ['x', 'a', 'b', 'a', 'b'])
            [(1, 2), (3, 4)]
            >>> find_sublist(['z'], ['x', 'y'])
            [(-1, -1)]
            """
            if not sub or all(token == "" for token in sub):
                return [(-1, -1)]

            sub_len = len(sub)
            matches = [
                (i, i + sub_len - 1)
                for i in range(len(lst) - sub_len + 1)
                if lst[i: i + sub_len] == sub
            ]
            return matches or [(-1, -1)]

        # Use zip() to ensure structure_labels, schemas, and task_types are aligned
        for schema_tokens, task_type, struct_label in zip(processed_schemas, task_types, structure_labels):
            if task_type != "classifications":
                # Get the corresponding structure label [count, [span1, span2, ...]]
                count, spans = struct_label
                transformed_spans = []
                for span in spans:
                    span_positions = []
                    for field in span:
                        # If the field is a list, process each element separately.
                        if isinstance(field, list):
                            nested_positions = []
                            for sub_field in field:
                                if str(sub_field).startswith("[selection]"):
                                    sub_field = sub_field[11:]
                                    field_tokens = [str(sub_field)]
                                    positions = find_sublist(field_tokens, text_tokens[:len_prefix])
                                else:
                                    field_tokens = self.tokenize_text(str(sub_field))
                                    positions = find_sublist(field_tokens, text_tokens)
                                nested_positions.extend(positions)
                            span_positions.append(nested_positions)
                        else:
                            if str(field).startswith("[selection]"):
                                field = field[11:]
                                field_tokens = [str(field)]
                            else:
                                field_tokens = self.tokenize_text(str(field))
                            pos = find_sublist(field_tokens, text_tokens)
                            span_positions.append(pos)
                    transformed_spans.append(span_positions)
                transformed_label = [count, transformed_spans]
                results.append({
                    "task_type": task_type,
                    "schema_tokens": schema_tokens,
                    "output": transformed_label
                })
            else:
                # For classification tasks, produce a boolean label list:
                # 1 for positive labels (if present in true_label), 0 for negative.
                classification_item = next(
                    (item for item in schema["classifications"] if schema_tokens[2].startswith(item["task"])
                     ), None
                )
                if classification_item is None:
                    raise ValueError(f"Missing classification item for task: {schema_tokens[2]}")

                bool_labels = [1 if label in classification_item["true_label"] else 0
                               for label in classification_item["labels"]]
                results.append({
                    "task_type": task_type,
                    "schema_tokens": schema_tokens,
                    "output": bool_labels
                })

        # Organize results into the expected dictionary structure
        schema_token_List = [schema["schema_tokens"] for schema in results]
        outputs = [schema["output"] for schema in results]
        task_types = [schema["task_type"] for schema in results]

        return {
            "schema_tokens_list": schema_token_List,
            "text_tokens": text_tokens,
            "outputs": outputs,
            "task_types": task_types,
            "start_token_idx_to_text_idx": start_token_idx_to_text_idx,
            "end_token_idx_to_text_idx": end_token_idx_to_text_idx
        }

    def format_input_with_mapping(self, schema_tokens_list: list, text_tokens: list) -> (dict, list):
        """
        Combines schema token lists and text tokens into a single token sequence,
        then tokenizes each token into subwords and produces a mapping for each subword.

        The combined token sequence is built by concatenating each schema token list (each representing
        a schema structure) separated by the special token [SEP_STRUCT], then appending the special token
        [SEP_TEXT] and finally the text tokens.

        For each subword token, a mapping tuple (seg_type, orig_idx, schema_idx) is produced where:
          - seg_type is:
               "schema" for tokens before [SEP_TEXT],
               "sep" for the [SEP_TEXT] token,
               "text" for tokens after [SEP_TEXT].
          - orig_idx is the index of the token in the combined_tokens list.
          - schema_idx is:
               the index of the schema structure (starting at 0) for tokens in the schema portion.
               For tokens after [SEP_TEXT] (the text part), schema_idx is set to len(schema_tokens_list) + 1.
               (Tokens corresponding to [SEP_TEXT] are also assigned this text schema index.)

        After building the subword list and mapping, tokens are converted to IDs and inputs for the model
        are prepared using the tokenizer's prepare_for_model function.

        Args:
            schema_tokens_list: A list of schema token lists (each representing one structure).
            text_tokens: A list of tokens representing the input text.

        Returns:
            A tuple (inputs, mapped_indices) where:
              - inputs: A dict of prepared inputs (including input_ids) for the model.
              - mapped_indices: A list of tuples (seg_type, orig_idx, schema_idx) for each subword token.
        """
        # Build combined tokens.
        combined_tokens = []
        for structure_tokens in schema_tokens_list:
            combined_tokens.extend(structure_tokens)
            combined_tokens.append(self.sep_struct)
        if combined_tokens:
            combined_tokens.pop()  # remove the trailing [SEP_STRUCT]
        combined_tokens.append(self.sep_text)
        combined_tokens = combined_tokens + text_tokens

        subword_list = []
        mapped_indices = []

        num_schemas = len(schema_tokens_list)
        text_schema_idx = num_schemas
        current_schema_idx = 0
        found_sep_text = False

        # Iterate over each token in combined_tokens and tokenize further into subwords.
        for orig_idx, token in enumerate(combined_tokens):
            if token == self.sep_text:
                seg_type = "sep"
                schema_index = text_schema_idx
                found_sep_text = True
            elif not found_sep_text:
                seg_type = "schema"
                schema_index = current_schema_idx
                if token == self.sep_struct:
                    current_schema_idx += 1
            else:
                seg_type = "text"
                schema_index = text_schema_idx

            sub_tokens = self.tokenizer.tokenize(token)
            subword_list.extend(sub_tokens)
            mapped_indices.extend([(seg_type, orig_idx, schema_index)] * len(sub_tokens))

        # Convert tokens to IDs.
        input_ids = self.tokenizer.convert_tokens_to_ids(subword_list)

        # Prepare inputs for the model.
        inputs = self.tokenizer.prepare_for_model(
            [input_ids],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        )

        return {
            "inputs": inputs,
            "mapped_indices": mapped_indices,
            "subword_list": subword_list
        }

    def extract_special_token_embeddings_per_schema(self, token_embeddings, input_ids, mapped_indices,
                                                    num_hierarchical_schemas):
        """
        Extract embeddings for special tokens [P], [C], [R], [L], [E] per hierarchical schema and
        collect the first subword embedding for each text token (i.e., tokens after [SEP_TEXT]).

        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, hidden_size).
            input_ids: Tensor of shape (batch_size, seq_len).
            mapped_indices: List of tuples (seg_type, orig_idx, schema_idx) for each subword token.
            num_hierarchical_schemas: Number of hierarchical schema segments.

        Returns:
            A dict with:
             - "special_embeddings": A list of lists (length=num_hierarchical_schemas) where each inner list
               contains embeddings (tensor slices) for special tokens in that schema.
             - "text_embeddings": A list of embeddings corresponding to the first subword of each text token.
        """
        input_ids_list = input_ids[0].tolist()
        # Initialize list for schema special tokens embeddings.
        collected_embeddings_per_schema = [[] for _ in range(num_hierarchical_schemas)]
        text_embeddings = []
        last_text_orig = None

        special_tokens_set = {self.p_token, self.c_token, self.e_token, self.r_token, self.l_token}

        # Iterate over each subword token in the sequence.
        # token_embeddings assumed shape: (1, seq_len, hidden_size)
        for i, tid in enumerate(input_ids_list):
            seg_type, orig_idx, schema_idx = mapped_indices[i]
            # Get the embedding for this subword token.
            embedding = token_embeddings[0, i, :]
            if seg_type == "schema":
                # Convert original token id to token string.
                token_str = self.tokenizer.convert_ids_to_tokens(tid)
                if token_str in special_tokens_set:
                    collected_embeddings_per_schema[schema_idx].append(embedding)
            elif seg_type == "text":
                # Only keep the first subword of a given text token.
                if last_text_orig != orig_idx:
                    text_embeddings.append(embedding)
                    last_text_orig = orig_idx

        return {
            "embs_per_schema": collected_embeddings_per_schema,
            "token_embeddings": torch.stack(text_embeddings)
        }

    def process_record(self, encoder, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single record by:
          1. Transforming the record to extract text tokens and non‐classification schema tokens.
          2. Formatting the input to obtain subword tokens and mapping indices.
          3. Passing the inputs through the encoder to get token embeddings.
          4. Extracting special token embeddings per schema and first subword embeddings for text tokens.

        Returns a dictionary with:
          - "inputs": The prepared input dictionary (including input_ids, attention_mask, etc.) for the model.
          - "mapped_indices": A list of mapping tuples (seg_type, orig_idx, schema_idx) for each subword token.
          - "special_embeddings": A list (one per schema) of embeddings for special tokens ([P], [C], [R], [L], [E]).
          - "text_embeddings": A list of embeddings corresponding to the first subword of each text token.
        """
        # 1. Transform the record.
        # (Assumes transform_single_record returns a dict with keys "text_tokens" and "schema_tokens_list")
        transformed = self.transform_single_record(record)
        text_tokens = transformed["text_tokens"]
        schema_tokens_list = transformed["schema_tokens_list"]

        # 2. Format input with mapping.
        inputs, mapped_indices, subword_list = self.format_input_with_mapping(schema_tokens_list, text_tokens).values()

        # to encoder device
        device = next(encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. Encode the inputs.
        outputs = encoder(**inputs)
        token_embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)

        # 4. Extract special token embeddings per schema.
        embs_per_schema, token_embeddings = self.extract_special_token_embeddings_per_schema(
            token_embeddings, inputs["input_ids"], mapped_indices, num_hierarchical_schemas=len(schema_tokens_list)
        ).values()

        return {
            "text_tokens": text_tokens,
            "inputs": inputs,
            "mapped_indices": mapped_indices,
            "embs_per_schema": embs_per_schema,
            "token_embeddings": token_embeddings,
            "structure_labels": transformed["outputs"],
            **transformed
        }
