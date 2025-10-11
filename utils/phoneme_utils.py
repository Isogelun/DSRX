import json
import pathlib
from typing import Dict, List, Union

from utils.hparams import hparams

PAD_INDEX = 0


class PhonemeDictionary:
    def __init__(
            self,
            dictionaries: Dict[str, pathlib.Path],
            extra_phonemes: List[str] = None,
            merged_groups: List[List[str]] = None
    ):
        """
        初始化音素词典，构建多语言音素索引系统。

        参数:
            dictionaries (Dict[str, pathlib.Path]): 语言名称到字典文件路径的映射。每个字典文件应为制表符分隔的文本文件，
                                                   每行格式为 "word phoneme1 phoneme2 ..."。
            extra_phonemes (List[str], optional): 额外添加的音素列表。支持使用 "language/phoneme" 格式指定语言特定音素。
                                                 默认为 None。
            merged_groups (List[List[str]], optional): 需要合并的音素组列表，每组中的音素将被视为等价。
                                                      支持使用 "language/phoneme" 格式指定语言特定音素。
                                                      默认为 None。

        异常:
            ValueError: 当音素标签格式错误、语言名称未识别或音素冲突时抛出。
        """
        # Step 1: Collect all phonemes
        all_phonemes = hparams.get('all_phonemes')
        if all_phonemes:
            all_phonemes = set(all_phonemes)
        else:
            all_phonemes = {'AP', 'SP'}

        lang_phoneme_separator = hparams.get('lang_phoneme_separator', '/')
        if isinstance(lang_phoneme_separator, (list, tuple, set)):
            separator = next(iter(lang_phoneme_separator))  # 取第一个作为主分隔符
        else:
            separator = lang_phoneme_separator
        self._separator = separator  # 保存到实例，供后续使用

        # Step 2: Parse extra phonemes

        if extra_phonemes:
            for ph in extra_phonemes:
                if self._separator in ph:
                    lang, name = ph.split(self._separator, maxsplit=1)
                    if lang not in dictionaries:
                        raise ValueError(
                            f"Invalid phoneme tag '{ph}' in extra phonemes: "
                            f"unrecognized language name '{lang}'."
                        )
                    if name in all_phonemes:
                        raise ValueError(
                            f"Invalid phoneme tag '{ph}' in extra phonemes: "
                            f"short name conflicts with existing tag."
                        )
                all_phonemes.add(ph)
        self._multi_langs = len(dictionaries) > 1
        for lang, dict_path in dictionaries.items():
            with open(dict_path, 'r', encoding='utf8') as dict_file:
                for line in dict_file:
                    _, phonemes = line.strip().split('\t')
                    phonemes = phonemes.split()
                    for phoneme in phonemes:
                        if self._separator in phoneme:
                            raise ValueError(
                                f"Invalid phoneme tag '{phoneme}' in dictionary '{dict_path}': "
                                f"should not contain the reserved character '{self._separator}'."
                            )
                        if phoneme in all_phonemes:
                            continue
                        if self._multi_langs:
                            all_phonemes.add(f'{lang}/{phoneme}')
                        else:
                            all_phonemes.add(phoneme)
        # Step 2: Parse merged phoneme groups
        if merged_groups is None:
            merged_groups = []
        else:
            _merged_groups = []
            for group in merged_groups:
                _group = []
                for phoneme in group:
                    if self._separator in phoneme:
                        lang, name = phoneme.split(self._separator, maxsplit=1)
                        if lang not in dictionaries:
                            raise ValueError(
                                f"Invalid phoneme tag '{phoneme}' in merged group: "
                                f"unrecognized language name '{lang}'."
                            )
                        if self._multi_langs:
                            element = phoneme
                        else:
                            element = name
                    else:
                        element = phoneme
                    if element not in all_phonemes:
                        raise ValueError(
                            f"Invalid phoneme tag '{phoneme}' in merged group: "
                            f"not found in phoneme set."
                        )
                    _group.append(element)
                _merged_groups.append(_group)
            merged_groups = [set(phones) for phones in _merged_groups if len(phones) > 1]
        # Step 3: Build phoneme index
        merged_phonemes_inverted_index = {}
        for idx, group in enumerate(merged_groups):
            other_idx = None
            for phoneme in group:
                if phoneme in merged_phonemes_inverted_index:
                    other_idx = merged_phonemes_inverted_index[phoneme]
                    break
            target_idx = idx if other_idx is None else other_idx
            for phoneme in group:
                merged_phonemes_inverted_index[phoneme] = target_idx
            if other_idx is not None:
                merged_groups[other_idx] |= group
                group.clear()
        phone_to_id = {}
        id_to_phone = []
        cross_lingual_phonemes = set()
        idx = 1
        for phoneme in sorted(all_phonemes):
            if phoneme in merged_phonemes_inverted_index:
                has_assigned = True
                for alias in merged_groups[merged_phonemes_inverted_index[phoneme]]:
                    if alias not in phone_to_id:
                        has_assigned = False
                        phone_to_id[alias] = idx
                if not has_assigned:
                    merged_group = sorted(merged_groups[merged_phonemes_inverted_index[phoneme]])
                    merged_from_langs = {
                        (alias.split(self._separator, maxsplit=1)[0] if self._separator in alias else None)
                        for alias in merged_group
                    }
                    id_to_phone.append(tuple(merged_group))
                    idx += 1
                    if len(merged_from_langs) > 1:
                        cross_lingual_phonemes.update(ph for ph in merged_group if self._separator in ph)
            else:
                phone_to_id[phoneme] = idx
                id_to_phone.append(phoneme)
                idx += 1
        self._phone_to_id: Dict[str, int] = phone_to_id
        self._id_to_phone: List[Union[str, tuple]] = id_to_phone
        self._cross_lingual_phonemes = frozenset(cross_lingual_phonemes)

    @property
    def vocab_size(self):
        return len(self._id_to_phone) + 1

    def __len__(self):
        return self.vocab_size

    @property
    def cross_lingual_phonemes(self):
        return self._cross_lingual_phonemes

    def is_cross_lingual(self, phone):
        return phone in self._cross_lingual_phonemes

    def encode_one(self, phone, lang=None):
        """
        将单个音素编码为ID。
        
        如果输入的音素已经包含语言标识（通过'/'分隔），则会拆分出语言和音素部分。
        如果是多语言模式且音素不在通用音素列表中，则会根据指定语言创建语言特定的音素标识。
        
        Args:
            phone (str): 音素字符串，可能包含语言前缀
            lang (str, optional): 语言标识，当phone中不包含语言信息时使用
            
        Returns:
            int: 对应音素在词典中的ID
        """
        # 如果音素中已包含语言标识，则拆分出语言和音素
        if self._separator in phone:
            lang, phone = phone.split(self._separator, maxsplit=1)
        # 如果不需要语言标识或音素已存在于词典中，直接返回ID
        if lang is None or not self._multi_langs or phone in self._phone_to_id:
            return self._phone_to_id[phone]
        # 如果是多语言模式且音素不包含语言标识，则添加语言前缀
        if self._separator not in phone:
            phone = f'{lang}{self._separator}{phone}'
        return self._phone_to_id[phone]

    def encode(self, sentence, lang=None):
        phones = sentence.strip().split() if isinstance(sentence, str) else sentence
        return [self.encode_one(phone, lang=lang) for phone in phones]

    def decode_one(self, idx, lang=None, scalar=True):
        if idx <= 0:
            return None
        phone = self._id_to_phone[idx - 1]
        if not scalar or isinstance(phone, str):
            return phone
        if lang is None or not self._multi_langs:
            return phone[0]
        for alias in phone:
            if alias.startswith(f'{lang}/'):
                return alias
        return phone[0]

    def decode(self, ids, lang=None, scalar=True):
        ids = list(ids)
        return ' '.join([
            self.decode_one(i, lang=lang, scalar=scalar)
            for i in ids
            if i >= 1
        ])

    def dump(self, filename):
        with open(filename, 'w', encoding='utf8') as fp:
            json.dump(self._phone_to_id, fp, ensure_ascii=False, indent=2)


_dictionary = None


def load_phoneme_dictionary() -> PhonemeDictionary:
    if _dictionary is not None:
        return _dictionary
    config_dicts = hparams.get('dictionaries')
    if config_dicts is not None:
        dicts = {}
        for lang, config_dict_path in config_dicts.items():
            dict_path = pathlib.Path(hparams['work_dir']) / f'dictionary-{lang}.txt'
            if not dict_path.exists():
                dict_path = pathlib.Path(config_dict_path)
            if not dict_path.exists():
                raise FileNotFoundError(
                    f"Could not locate dictionary for language '{lang}'."
                )
            dicts[lang] = dict_path
    else:
        dict_path = pathlib.Path(hparams['work_dir']) / 'dictionary.txt'
        if not dict_path.exists():
            dict_path = pathlib.Path(hparams['dictionary'])
        if not dict_path.exists():
            raise FileNotFoundError(
                f"Could not locate dictionary file."
            )
        dicts = {
            'default': dict_path
        }
    return PhonemeDictionary(
        dictionaries=dicts,
        extra_phonemes=hparams.get('extra_phonemes'),
        merged_groups=hparams.get('merged_phoneme_groups')
    )
