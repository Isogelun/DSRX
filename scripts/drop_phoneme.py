import argparse
import json
import pathlib
import re
from typing import Dict, List

import torch

ID_LIST_PATTERN = r'(\d+)?(,\d+)*,?'
PHONEME_EMBED_SUFFIX = 'txt_embed.weight'

def _parse_id_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(',') if part.strip()]

def _parse_name_list(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(',') if part.strip()]

def _load_dictionary(path: pathlib.Path) -> Dict[str, int]:
    with path.open('r', encoding='utf8') as dict_file:
        data = json.load(dict_file)
    if not isinstance(data, dict):
        raise ValueError('Dictionary file must contain a JSON object mapping phoneme tokens to ids.')
    mapping = {}
    for token, idx in data.items():
        try:
            mapping[token] = int(idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid id for phoneme '{token}': {idx}") from exc
    return mapping

def _apply_fill(tensor: torch.Tensor, drop_ids: List[int], fill: str):
    if not drop_ids:
        return False
    num_embeddings, hidden_size = tensor.shape
    drop_tensor = torch.tensor(drop_ids, dtype=torch.long, device=tensor.device)
    if fill == 'zeros':
        tensor.index_fill_(0, drop_tensor, 0.0)
    elif fill == 'random':
        tensor[drop_tensor] = torch.randn(
            (len(drop_ids), hidden_size),
            dtype=tensor.dtype,
            device=tensor.device
        )
    elif fill == 'mean':
        mean_vec = tensor.mean(dim=0, keepdim=True).clone()
        tensor[drop_tensor] = mean_vec
    elif fill == 'cyclic':
        retain_ids = sorted(set(range(num_embeddings)) - set(drop_ids))
        if not retain_ids:
            raise ValueError('Cannot use cyclic fill when all phoneme embeddings are dropped.')
        source_rows = torch.stack([
            tensor[retain_ids[i % len(retain_ids)]].clone()
            for i in range(len(drop_ids))
        ], dim=0)
        tensor[drop_tensor] = source_rows
    else:
        raise ValueError(f'Unknown fill method: {fill}')
    return True


def main():
    parser = argparse.ArgumentParser(description='Drop phoneme embeddings from a checkpoint.')
    parser.add_argument('input', type=str, help='Path to the input checkpoint file.')
    parser.add_argument('output', type=str, help='Path to the output checkpoint file.')
    parser.add_argument('--drop-ids', type=str, help='Comma separated phoneme ids to drop.')
    parser.add_argument('--drop-names', type=str, help='Comma separated phoneme tokens to drop.')
    parser.add_argument('--dictionary', type=str,
                        help='Path to a phoneme dictionary JSON (e.g., exported \'*.phonemes.json\') '
                             'used to resolve phoneme tokens to ids when --drop-names is provided.')
    parser.add_argument('--fill', type=str, default='zeros', choices=['zeros', 'random', 'mean', 'cyclic'],
                        help='Filling strategy for dropped embeddings.')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite the output file if it already exists.')
    args = parser.parse_args()

    if not args.drop_ids and not args.drop_names:
        parser.error('At least one of --drop-ids or --drop-names must be specified.')

    if args.drop_ids and not re.fullmatch(ID_LIST_PATTERN, args.drop_ids):
        print(f"Invalid format for --drop-ids: '{args.drop_ids}'")
        return

    drop_ids = set()
    if args.drop_ids:
        drop_ids.update(_parse_id_list(args.drop_ids))

    if args.drop_names:
        if not args.dictionary:
            parser.error('--drop-names requires --dictionary to resolve phoneme tokens to ids.')
        dictionary_path = pathlib.Path(args.dictionary).resolve()
        if not dictionary_path.exists():
            parser.error(f"Dictionary file does not exist: {dictionary_path}")
        token_to_id = _load_dictionary(dictionary_path)
        missing_tokens = []
        for token in _parse_name_list(args.drop_names):
            if token in token_to_id:
                drop_ids.add(token_to_id[token])
            else:
                missing_tokens.append(token)
        if missing_tokens:
            print(f"| warning: phoneme tokens not found in dictionary: {missing_tokens}")

    if not drop_ids:
        print('| info: no phoneme ids resolved from the provided arguments; aborting without changes.')
        return

    input_ckpt = pathlib.Path(args.input).resolve()
    output_ckpt = pathlib.Path(args.output).resolve()
    assert input_ckpt.exists(), 'The input file does not exist.'
    assert args.overwrite or not output_ckpt.exists(), \
        'The output file already exists or is the same as the input file.\n' \
        'This is not recommended because embedding dropping scripts may not be stable, ' \
        'and you may be at risk of losing your model.\n' \
        'If you are sure to OVERWRITE the existing file, please re-run this script with the \'--overwrite\' argument.'

    ckpt_loaded = torch.load(input_ckpt, map_location='cpu')
    if 'state_dict' not in ckpt_loaded:
        raise KeyError("The checkpoint does not contain a 'state_dict' entry.")
    state_dict = ckpt_loaded['state_dict']

    target_keys = [
        key for key in state_dict
        if key.endswith(PHONEME_EMBED_SUFFIX) and isinstance(state_dict[key], torch.Tensor)
    ]

    if not target_keys:
        print('| warning: no phoneme embedding tensors were found in the checkpoint. Nothing was changed.')
        torch.save(ckpt_loaded, output_ckpt)
        return

    sorted_drop_ids = sorted(drop_ids)
    invalid_ids = set()
    applied_ids = set()
    modified_keys = []
    for key in target_keys:
        tensor = state_dict[key]
        if tensor.ndim != 2:
            continue
        num_embeddings = tensor.shape[0]
        valid_ids = [idx for idx in sorted_drop_ids if 0 <= idx < num_embeddings]
        invalid_ids.update(idx for idx in sorted_drop_ids if idx < 0 or idx >= num_embeddings)
        if not valid_ids:
            continue
        _apply_fill(tensor, valid_ids, args.fill)
        modified_keys.append(key)
        applied_ids.update(valid_ids)

    if invalid_ids:
        print(f"| warning: ignoring phoneme ids out of range: {sorted(invalid_ids)}")

    if not modified_keys:
        print('| info: no phoneme embeddings matched the requested ids; no changes written.')
        torch.save(ckpt_loaded, output_ckpt)
        return

    applied_ids_sorted = sorted(applied_ids)
    print(f"| info: dropped {len(applied_ids_sorted)} phoneme ids (ids={applied_ids_sorted}) across {len(modified_keys)} tensor(s) with fill='{args.fill}'.")
    torch.save(ckpt_loaded, output_ckpt)


if __name__ == '__main__':
    main()
