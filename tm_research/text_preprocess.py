"""Shared Vietnamese text preprocessing pipeline.

Extracted from DataPreprocessing.ipynb (§2 – §3) so both VSMEC and VSFC
notebooks use identical cleaning logic without dictionary drift.

Public API
----------
    preprocess_vietnamese_text(text: str) -> str
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

EMOJI_TO_VIETNAMESE = {
    '😀': 'vui vẻ', '😃': 'vui vẻ', '😄': 'vui vẻ', '😁': 'vui vẻ', '😆': 'vui vẻ',
    '😅': 'vui vẻ', '🤣': 'cười lớn', '😂': 'cười lớn', '🙂': 'mỉm cười', '🙃': 'mỉm cười',
    '😉': 'nháy mắt', '😊': 'hạnh phúc', '😇': 'thiên thần', '🥰': 'yêu thương',
    '😍': 'yêu thích', '🤩': 'ngưỡng mộ', '😘': 'hôn', '😗': 'hôn', '☺️': 'hạnh phúc',
    '😚': 'hôn', '😙': 'hôn', '🥲': 'vui buồn lẫn lộn',
    '❤️': 'yêu thương', '💕': 'yêu thương', '💗': 'yêu thương', '💖': 'yêu thương',
    '💘': 'yêu thương', '💝': 'yêu thương', '💞': 'yêu thương', '💓': 'yêu thương',
    '💜': 'yêu thương', '💙': 'yêu thương', '💚': 'yêu thương', '🧡': 'yêu thương',
    '🖤': 'yêu thương', '🤍': 'yêu thương', '🤎': 'yêu thương', '💛': 'yêu thương',
    '😢': 'buồn bã', '😭': 'khóc', '😿': 'buồn bã', '😞': 'thất vọng',
    '😔': 'buồn bã', '😟': 'lo lắng', '😕': 'bối rối', '🙁': 'buồn bã',
    '☹️': 'buồn bã', '😣': 'đau khổ', '😖': 'khó chịu', '😫': 'mệt mỏi',
    '😩': 'mệt mỏi', '🥺': 'van xin', '😪': 'buồn ngủ', '😥': 'lo lắng',
    '😠': 'tức giận', '😡': 'tức giận', '🤬': 'tức giận', '😤': 'bực tức',
    '💢': 'tức giận', '👿': 'tức giận', '😈': 'tinh nghịch',
    '😨': 'sợ hãi', '😰': 'lo sợ', '😱': 'kinh hãi', '🥶': 'lạnh',
    '😮': 'ngạc nhiên', '😯': 'ngạc nhiên', '😲': 'sửng sốt', '😳': 'xấu hổ',
    '🤯': 'choáng váng', '😬': 'căng thẳng', '🙀': 'sợ hãi',
    '🤢': 'buồn nôn', '🤮': 'ghê tởm', '😷': 'bệnh', '🤧': 'hắt xì',
    '🤒': 'ốm', '🤕': 'đau', '😵': 'chóng mặt', '🥴': 'say',
    '🤔': 'suy nghĩ', '🤨': 'nghi ngờ', '😐': 'bình thường', '😑': 'bình thường',
    '😶': 'im lặng', '🙄': 'chán', '😏': 'ranh mãnh', '😒': 'không hài lòng',
    '🤗': 'ôm', '🤭': 'che miệng cười', '🤫': 'im lặng', '🤥': 'nói dối',
    '😴': 'ngủ', '💤': 'ngủ', '👍': 'tốt', '👎': 'tệ', '👌': 'được',
    '✌️': 'chiến thắng', '🤞': 'chúc may mắn', '🙏': 'cầu nguyện',
    '💪': 'mạnh mẽ', '🔥': 'nóng bỏng', '✨': 'lấp lánh', '🌟': 'ngôi sao',
    '💯': 'hoàn hảo', '😎': 'ngầu', '🤓': 'thông minh', '🥳': 'ăn mừng',
    '😋': 'ngon', '🤤': 'thèm', '😜': 'tinh nghịch', '😝': 'tinh nghịch',
    '😛': 'lè lưỡi', '🤪': 'điên', '👀': 'nhìn', '👁️': 'nhìn',
}

TEXT_EMOTICONS = {
    ':)': 'vui vẻ', ':-)': 'vui vẻ', '(:': 'vui vẻ',
    ':D': 'cười lớn', ':-D': 'cười lớn',
    ':))': 'cười lớn', ':)))': 'cười lớn', ':))))': 'cười lớn',
    '=))': 'cười lớn', '=)))': 'cười lớn',
    ':3': 'dễ thương',
    ':(': 'buồn bã', ':-(': 'buồn bã', ":'(": 'khóc',
    ':((': 'buồn bã', ':(((': 'rất buồn',
    ':P': 'tinh nghịch', ':-P': 'tinh nghịch', ':p': 'tinh nghịch',
    ';)': 'nháy mắt', ';-)': 'nháy mắt',
    ':O': 'ngạc nhiên', ':-O': 'ngạc nhiên', ':o': 'ngạc nhiên',
    'o.O': 'ngạc nhiên', 'O.o': 'ngạc nhiên',
    '@@': 'choáng',
    '-_-': 'chán', '-.-': 'chán',
    '>.<': 'khó chịu',
    '^_^': 'vui vẻ', '^.^': 'vui vẻ',
    'T_T': 'khóc', 'T.T': 'khóc',
    '>_<': 'khó chịu',
    '<3': 'yêu thương', '</3': 'tan vỡ',
    ':*': 'hôn', ':-*': 'hôn',
}

ABBREVIATIONS = {
    'ko': 'không', 'k': 'không', 'hok': 'không', 'hem': 'không', 'hông': 'không',
    'dc': 'được', 'đc': 'được', 'đk': 'được', 'dk': 'được', 'đuoc': 'được',
    'vs': 'với', 'voi': 'với', 'v': 'với',
    'mk': 'mình', 'mik': 'mình', 'minh': 'mình',
    'bn': 'bạn', 'pn': 'bạn', 'bạng': 'bạn',
    'cx': 'cũng', 'cg': 'cũng', 'cug': 'cũng',
    'ntn': 'như thế nào', 'sn': 'sao',
    'bh': 'bây giờ', 'bjo': 'bây giờ', 'bjờ': 'bây giờ', 'bgiờ': 'bây giờ',
    'j': 'gì', 'ji': 'gì', 'z': 'vậy', 'zậy': 'vậy', 'r': 'rồi',
    'trc': 'trước', 'trog': 'trong', 'tg': 'thời gian',
    'nt': 'nhắn tin', 'nv': 'nhân vật',
    'ns': 'nói', 'nc': 'nói chuyện', 'nch': 'nói chuyện',
    'bt': 'bình thường', 'bth': 'bình thường', 'binh thuong': 'bình thường',
    'thg': 'thằng', 'thag': 'thằng',
    'ng': 'người', 'ngta': 'người ta', 'nguoi': 'người',
    'ak': 'à', 'ạk': 'ạ', 'nka': 'nhé', 'nhe': 'nhé',
    'gđ': 'gia đình', 'gd': 'gia đình',
    'hj': 'hihi', 'haha': 'haha', 'hihi': 'hihi', 'huhu': 'huhu',
    'kkk': 'haha', 'kk': 'haha', 'wkwk': 'haha',
    'vkl': 'rất', 'vl': 'rất', 'vcl': 'rất',
    'clgt': 'chắc luôn', 'cl': 'chắc luôn',
    'tl': 'trả lời', 'rep': 'trả lời',
    'acc': 'tài khoản', 'fb': 'facebook', 'ib': 'nhắn tin',
    'đt': 'điện thoại', 'dt': 'điện thoại',
    'tv': 'tivi', 'tp': 'thành phố', 'tphcm': 'thành phố hồ chí minh',
    'hn': 'hà nội', 'sg': 'sài gòn',
    'trg': 'trường', 'truong': 'trường',
    'lm': 'làm', 'lam': 'làm',
    'cty': 'công ty', 'cong ty': 'công ty',
    'thang': 'tháng',
    'qtr': 'quan trọng', 'qtrog': 'quan trọng',
    'sp': 'sản phẩm', 'dv': 'dịch vụ',
    'hq': 'hàn quốc', 'tq': 'trung quốc',
    'thik': 'thích', 'tk': 'tài khoản',
    't': 'tao',
}

# ---------------------------------------------------------------------------
# Compiled regexes (module-level for performance)
# ---------------------------------------------------------------------------

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

_VN_CHARS_RE = re.compile(
    r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ.,!?]',
    flags=re.IGNORECASE,
)

_ELONGATED_CHAR_RE = re.compile(r'([^\W\d_])\1{2,}', flags=re.UNICODE)

_SORTED_EMOTICONS = sorted(TEXT_EMOTICONS.items(), key=lambda x: len(x[0]), reverse=True)

# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def replace_emojis(text: str) -> str:
    for emoji_char, descriptor in EMOJI_TO_VIETNAMESE.items():
        text = text.replace(emoji_char, f' {descriptor} ')
    return _EMOJI_RE.sub(' ', text)


def replace_text_emoticons(text: str) -> str:
    for emoticon, descriptor in _SORTED_EMOTICONS:
        escaped = re.escape(emoticon)
        text = re.sub(escaped, f' {descriptor} ', text)
    return text


def replace_abbreviations(text: str) -> str:
    words = text.split()
    return ' '.join(ABBREVIATIONS.get(w.lower(), w) for w in words)


def normalize_elongated_characters(text: str) -> str:
    return _ELONGATED_CHAR_RE.sub(r'\1', text)


def remove_numbers(text: str) -> str:
    return ' '.join(re.sub(r'\d+', ' ', text).split())


def clean_special_characters(text: str) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = _VN_CHARS_RE.sub(' ', text)
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_vietnamese_text(text: str) -> str:
    """Full cleaning pipeline for Vietnamese social-media text.

    Steps (in order):
      1. Emoji → Vietnamese descriptor words
      2. Text emoticons (e.g. :)) ) → descriptor words
      3. Abbreviation expansion
      4. Elongated character normalisation (soooo → so)
      5. Number removal
      6. URL / HTML / special character removal + whitespace normalisation
      7. Lowercase

    Stopwords are intentionally NOT removed — BERT models need negation
    words like *không* and degree adverbs like *rất*.
    """
    if not isinstance(text, str):
        return text
    text = replace_emojis(text)
    text = replace_text_emoticons(text)
    text = replace_abbreviations(text)
    text = normalize_elongated_characters(text)
    text = remove_numbers(text)
    text = clean_special_characters(text)
    return text.lower().strip() if text else text
