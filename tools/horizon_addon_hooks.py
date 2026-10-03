#!/usr/bin/env python3
"""Validate local Horizon addon hooks before adding their address-only metadata.

No login, game launch or retail bytes in output. Requires pefile/capstone and the
local primary Ashita pointer file. The original image hashes, unique patterns,
known function boundaries, calling convention and chat forwarding are checked.
"""
import argparse
import configparser
import hashlib
import json
from pathlib import Path
import re
import struct
import capstone
import pefile

ROOT = Path(__file__).resolve().parents[1]
RETAIL = 'bda769e226d71a43335c105fd6f72ed19af0a3d815a079b367356de0731a0d9a'
EXPECTED = {'parse_input': 0x10080bd0, 'write_line': 0x100765d0,
            'packet_decrypt': 0x100dd400, 'packet_encrypt': 0x100dd2d0}


def validate(image_path, retail_path, pointers_path):
    retail_digest = hashlib.sha256(retail_path.read_bytes()).hexdigest()
    if retail_digest != RETAIL:
        raise ValueError('retail build identity mismatch')
    pe = pefile.PE(str(image_path))
    text = next(s for s in pe.sections if s.Name.rstrip(b'\0') == b'.text')
    base = pe.OPTIONAL_HEADER.ImageBase
    start = base + text.VirtualAddress
    raw = text.get_data()[:text.Misc_VirtualSize]
    metadata = json.loads((ROOT / 'meta/FFXiMain.2025-11-12.meta.json').read_text())
    functions = {f['entry']: f for f in metadata['functions']}
    pointers = configparser.ConfigParser(interpolation=None)
    pointers.read(pointers_path)

    def unique(section):
        source = pointers[section]
        if source['module'].strip().lower() != 'ffximain.dll':
            raise ValueError('primary pointer module differs')
        pattern = source['pattern'].strip()
        expr = b''.join(b'.' if pattern[i:i+2] == '??' else
                        re.escape(bytes.fromhex(pattern[i:i+2]))
                        for i in range(0, len(pattern), 2))
        hits = [start + m.start() + int(source['offset'])
                for m in re.finditer(expr, raw, re.DOTALL)]
        if len(hits) != 1:
            raise ValueError(f'{section}: exactly one pattern match required')
        return hits[0]

    wraps = {'parse_input': unique('chat.parseinputtext'),
             'packet_decrypt': unique('packets.decryptbuffer'),
             'packet_encrypt': unique('packets.encryptbuffer')}
    chat_method = unique('chat.addchatmessage')
    dis = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    dis.detail = True
    writers = []
    # Find the cdecl mode/text wrapper from shape, not a copied newer address.
    for entry, function in functions.items():
        if len(function['ranges']) != 1 or function['ranges'][0][1] - entry != 44:
            continue
        code = list(dis.disasm(raw[entry-start:entry-start+44], entry))
        if len(code) != 15:
            continue
        if [(i.mnemonic, i.op_str) for i in code[:7]] != [
                ('mov', 'al, byte ptr [esp + 4]'),
                ('mov', 'edx, dword ptr [esp + 8]'),
                ('mov', 'byte ptr [esp + 4], al'), ('xor', 'eax, eax'),
                ('push', 'eax'), ('push', '1'), ('lea', 'ecx, [esp + 0xc]')]:
            continue
        if code[9].address != entry + 0x17 or code[9].mnemonic != 'mov' or not code[9].op_str.startswith('ecx, dword ptr [0x'):
            continue
        if code[-2].mnemonic != 'call' or int(code[-2].op_str, 16) != chat_method:
            continue
        if code[-1].mnemonic != 'ret' or code[-1].op_str:
            continue
        writers.append((entry, code[9].operands[1].mem.disp))
    if len(writers) != 1:
        raise ValueError('exactly one cdecl two-argument chat wrapper required')
    wraps['write_line'], chat_global = writers[0]
    if wraps != EXPECTED:
        raise ValueError('resolved anchors differ from independently reviewed Horizon metadata')
    checks = []
    for name, entry in wraps.items():
        function = functions.get(entry)
        if not function or function['thunk'] or function['ranges'][0][0] != entry:
            raise ValueError(f'{name}: known nonthunk function entry required')
        size = sum(end-begin for begin, end in function['ranges'])
        code = []
        for begin, end in function['ranges']:
            code += list(dis.disasm(raw[begin-start:end-start], begin))
        returns = [i for i in code if i.mnemonic == 'ret']
        if not returns or any(i.op_str for i in returns):
            raise ValueError(f'{name}: cdecl stack cleanup required')
        checks.append({'name': name, 'entry': f'0x{entry:08x}',
                       'function_bytes': size, 'known_function_entry': True,
                       'cdecl_returns': len(returns), 'source_pattern_unique': True})
    return {'retail_sha256': retail_digest,
            'unpacked_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'pointer_source_sha256': hashlib.sha256(pointers_path.read_bytes()).hexdigest(),
            'wraps': {key: f'0x{value:08x}' for key, value in wraps.items()},
            'chat_method': f'0x{chat_method:08x}', 'chat_global': f'0x{chat_global:08x}',
            'checks': checks, 'static_only': True,
            'runtime_chat_packet_addon_admitted': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, default=ROOT/'generated/images/2025-11-12/FFXiMain.unpacked.dll')
    p.add_argument('--retail', type=Path, default=ROOT/'generated/images/2025-11-12/FFXiMain.retail.dll')
    p.add_argument('--pointers', type=Path, required=True)
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    result = validate(args.image, args.retail, args.pointers)
    if args.write:
        catalog = ROOT/'meta/builds.json'
        original = json.loads(catalog.read_text())
        entry = original['builds']['2025-11-12']
        if entry['FFXiMain.dll']['sha256'] != RETAIL or entry.get('wraps', {}) not in [{}, result['wraps']]:
            raise ValueError('existing catalog identity or nonempty hooks differ')
        entry['wraps'] = result['wraps']
        catalog.write_text(json.dumps(original, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
