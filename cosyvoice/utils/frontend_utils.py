# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import regex
chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
cyrillic_char_pattern = re.compile(r'[\u0400-\u04ff]+')


# whether contain chinese character
def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


# whether contain cyrillic (Russian, etc.) character
def contains_cyrillic(text):
    return bool(cyrillic_char_pattern.search(text))


# replace special symbol
def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


# spell Arabic numerals
def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


# split paragrah logic：
# 1. per sentence max len token_max_n, min len token_min_n, merge if last sentence len less than merge_len
# 2. cal sentence len according to lang
# 3. split sentence according to puncatation
def split_paragraph(text: str, tokenize, lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False):
    # For Chinese, use character count (1 char ≈ 1 token)
    # For other languages (en, ru), use actual token count from tokenizer
    use_char_count = lang == "zh"
    
    def calc_utt_length(_text: str):
        if use_char_count:
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        if use_char_count:
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    # Punctuation for sentence splitting
    # Include ellipsis (…) as sentence delimiter - common in Russian texts
    if lang == "zh":
        pounc = ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';', '…']
    elif lang == "ru":
        # Russian punctuation
        pounc = ['.', '?', '!', ';', ':', '…']
    else:
        pounc = ['.', '?', '!', ';', ':', '…']
    if comma_split:
        pounc.extend(['，', ','])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."
    
    # Handle three dots "..." as single delimiter by replacing with ellipsis
    text = text.replace("...", "…")

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st: i]) > 0:
                utts.append(text[st: i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', '”']:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if calc_utt_length(cur_utt + utt) > token_max_n and calc_utt_length(cur_utt) > token_min_n:
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


# remove blank between chinese character
def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))


# ============================================================================
# Stress marks processing for Russian text
# ============================================================================

RUSSIAN_VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')
COMBINING_ACUTE = '\u0301'


def convert_stress_marks(text: str) -> str:
    """
    Converts silero-stress format (+before vowel) to Unicode (vowel+U+0301).
    
    Example: "за+мок" -> "за́мок"
    
    Args:
        text: Text with stress marks in silero-stress format
    
    Returns:
        Text with stress marks in Unicode combining acute accent format
    """
    result = []
    i = 0
    while i < len(text):
        if text[i] == '+' and i + 1 < len(text) and text[i + 1] in RUSSIAN_VOWELS:
            # + before vowel -> vowel + combining acute accent
            result.append(text[i + 1])
            result.append(COMBINING_ACUTE)
            i += 2
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def split_text_smart(text: str, max_chars: int = 400) -> list:
    """
    Splits text into optimal-sized chunks without breaking words or sentences.
    
    Optimal chunk size for silero-stress processing:
    - 300 chars: maximum speed ~14,800 chars/sec
    - 400 chars: good balance ~11,600 chars/sec
    
    Split priority:
    1. At sentence boundary (. ! ? ; :)
    2. At comma or other punctuation marks
    3. At space between words
    
    Args:
        text: Input text
        max_chars: Maximum chunk size (default 400)
    
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_chars:
        return [text] if text else []
    
    chunks = []
    current_pos = 0
    text_len = len(text)
    
    # Priority delimiters
    sentence_delims = '.!?;:。？！；：'
    clause_delims = ',—–-，、'
    
    while current_pos < text_len:
        # If remaining text is less than max_chars, take it all
        if current_pos + max_chars >= text_len:
            chunk = text[current_pos:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        # Find best split point within max_chars
        chunk_end = current_pos + max_chars
        best_split = None
        
        # 1. Look for sentence end (priority) - iterate from end to start
        for i in range(chunk_end, current_pos, -1):
            if i > 0 and text[i - 1] in sentence_delims:
                # Check that after delimiter there's space or end of text
                if i >= text_len or text[i].isspace() or text[i] in '"»"\'':
                    best_split = i
                    break
        
        # 2. If not found, look for comma or dash (minimum half chunk)
        if best_split is None:
            min_pos = current_pos + max_chars // 2
            for i in range(chunk_end, min_pos, -1):
                if i > 0 and text[i - 1] in clause_delims:
                    if i >= text_len or text[i].isspace():
                        best_split = i
                        break
        
        # 3. If not found, look for space
        if best_split is None:
            min_pos = current_pos + max_chars // 2
            for i in range(chunk_end, min_pos, -1):
                if text[i].isspace():
                    best_split = i + 1
                    break
        
        # 4. Fallback - hard split (should not happen in normal text)
        if best_split is None:
            best_split = chunk_end
        
        chunk = text[current_pos:best_split].strip()
        if chunk:
            chunks.append(chunk)
        current_pos = best_split
        
        # Skip leading spaces of next chunk
        while current_pos < text_len and text[current_pos].isspace():
            current_pos += 1
    
    return chunks
