import os
import json
import unicodedata
import morfessor

# Script groups based on the paper
SCRIPT_GROUPS = {
    0: ['Latin'],
    1: ['Latin'], # 1 is also for Latin
    2: ['Greek', 'Cyrillic', 'Armenian', 'Georgian'], # Non-Latin Alphabetic
    3: ['Hebrew', 'Arabic', 'Syriac', 'Thaana', 'Tifinagh'], # Abjads
    4: ['Devanagari', 'Bengali', 'Gurmukhi', 'Gujarati', 'Oriya', 'Sinhala', 'Tibetan', 'Ethiopic', 'Cherokee', 'Canadian_Aboriginal', 'Ogham', 'Runic', 'Mongolian'], # Abugidas North + some others that were grouped
    5: ['Telugu', 'Kannada', 'Tamil', 'Malayalam', 'Thai', 'Lao', 'Myanmar', 'Tai', 'Tagalog', 'Khmer'], # Abugidas South
    6: ['Hangul', 'Han', 'Yi', 'Katakana', 'Hiragana', 'Bopomofo'], # CJK
    7: [] # Other (Remaining scripts)
}

# Add mixed, common etc grouping to group 1
GROUP_1_SPECIAL = ['Common', 'Unknown', 'Inherited']

def get_script(char):
    try:
        name = unicodedata.name(char).split()[0]
        if name == 'LATIN': return 'Latin'
        if name == 'GREEK': return 'Greek'
        if name == 'CYRILLIC': return 'Cyrillic'
        if name == 'ARMENIAN': return 'Armenian'
        if name == 'GEORGIAN': return 'Georgian'
        if name == 'HEBREW': return 'Hebrew'
        if name == 'ARABIC': return 'Arabic'
        if name == 'SYRIAC': return 'Syriac'
        if name == 'THAANA': return 'Thaana'
        if name == 'DEVANAGARI': return 'Devanagari'
        if name == 'BENGALI': return 'Bengali'
        if name == 'GURMUKHI': return 'Gurmukhi'
        if name == 'GUJARATI': return 'Gujarati'
        if name == 'ORIYA': return 'Oriya'
        if name == 'TAMIL': return 'Tamil'
        if name == 'TELUGU': return 'Telugu'
        if name == 'KANNADA': return 'Kannada'
        if name == 'MALAYALAM': return 'Malayalam'
        if name == 'SINHALA': return 'Sinhala'
        if name == 'THAI': return 'Thai'
        if name == 'LAO': return 'Lao'
        if name == 'TIBETAN': return 'Tibetan'
        if name == 'MYANMAR': return 'Myanmar'
        if name == 'MONGOLIAN': return 'Mongolian'
        if name == 'HANGUL': return 'Hangul'
        if name == 'HIRAGANA': return 'Hiragana'
        if name == 'KATAKANA': return 'Katakana'
        if name == 'BOPOMOFO': return 'Bopomofo'
        if name == 'CJK': return 'Han'
        return 'Unknown'
    except ValueError:
        return 'Unknown'

def get_script_group(morpheme):
    scripts = set(get_script(c) for c in morpheme)
    if len(scripts) > 1:
        return 1 # Mixed
    
    script = list(scripts)[0]
    if script in GROUP_1_SPECIAL:
        return 1
    
    for group_id, script_list in SCRIPT_GROUPS.items():
        if script in script_list:
            if group_id == 1 and script == 'Latin':
                return 0 # latin goes to 0 or 1
            return group_id
    return 7 # Other

class MyteTokenizer:
    def __init__(self):
        self.morfessor_model = None
        self.morpheme_byte_map = {}
        self.byte_morpheme_map = {}
        
        # Freed leading bytes for each group (from paper Table 1)
        self.group_leading_bytes = {
            0: [0x42, 0x4A, 0x52], # Latin
            1: [0x43, 0x4B, 0x53], # Mixed, Common
            2: [0x44, 0x4C, 0x54], # Non-Latin Alphabetic
            3: [0x45, 0x4D, 0x55], # Abjads
            4: [0x46, 0x4E, 0x56], # Abugidas North
            5: [0x47, 0x4F, 0x57], # Abugidas South
            6: [0x48, 0x50, 0x58], # CJK
            7: [0x49, 0x51, 0x59]  # Other
        }
        
        # Continuation bytes
        self.continuation_bytes = list(range(0x80, 0xC0)) # 80 to BF
        
        self.capitalization_marker_byte = bytes([0x41])
        self.capitalization_marker_char = '\uE000' # Private Use Area character for internal processing

    def _normalize(self, text):
        res = []
        for char in text:
            if char.isupper():
                res.append(self.capitalization_marker_char)
                res.append(char.lower())
            else:
                res.append(char)
        return "".join(res)
    
    def _assign_bytes(self, ranked_morphemes):
        # ranked_morphemes is a list of (morpheme, score) sorted by score descending
        
        group_counts = {i: 0 for i in range(8)}
        
        for morpheme, score in ranked_morphemes:
            if morpheme == self.capitalization_marker_char:
                # Already handled, not a true morpheme discovered by Morfessor
                continue
            
            group = get_script_group(morpheme)
            count = group_counts[group]
            
            # Select prefix bytes based on count within group
            leading_byte_idx = (count // (64 * 64)) % 3 # simplistic allocation
            leading_byte = self.group_leading_bytes[group][leading_byte_idx]
            
            if count < 64:
                # 2-byte code
                cb1 = self.continuation_bytes[count]
                byte_code = bytes([leading_byte, cb1])
            elif count < 64 + 64*64:
                # 3-byte code
                idx = count - 64
                cb1 = self.continuation_bytes[idx // 64]
                cb2 = self.continuation_bytes[idx % 64]
                byte_code = bytes([leading_byte, cb1, cb2])
            else:
                # 4-byte code (up to 262,144)
                idx = count - 64 - 64*64
                cb1 = self.continuation_bytes[idx // (64*64)]
                cb2 = self.continuation_bytes[(idx // 64) % 64]
                cb3 = self.continuation_bytes[idx % 64]
                byte_code = bytes([leading_byte, cb1, cb2, cb3])
                
            self.morpheme_byte_map[morpheme] = byte_code
            self.byte_morpheme_map[byte_code] = morpheme
            
            group_counts[group] += 1
            
    def train(self, corpus_file, vocab_size, output_dir, lang):
        print(f"Training Morfessor on {corpus_file}...")
        
        # Prepare normalized corpus
        temp_corpus = os.path.join(output_dir, f"temp_{lang}_normalized.txt")
        with open(corpus_file, 'r', encoding='utf-8') as fin, \
             open(temp_corpus, 'w', encoding='utf-8') as fout:
            for line in fin:
                fout.write(self._normalize(line))
                
        io = morfessor.MorfessorIO()
        train_data = list(io.read_corpus_file(temp_corpus))
        
        self.morfessor_model = morfessor.BaselineModel()
        self.morfessor_model.load_data(train_data)
        self.morfessor_model.train_batch()
        
        print(f"Morfessor trained. Extracting top {vocab_size} morphemes based on count/score...")
        # Morfessor's get_constructions gives us the learned morphemes and their counts/scores
        constructions = self.morfessor_model.get_constructions()
        
        # Sort by count (which roughly correlates to hypothetical loss reduction in baseline model)
        # More sophisticated extraction would evaluate hypothetical loss reduction for each
        ranked_morphemes = sorted(constructions, key=lambda x: x[1], reverse=True)[:vocab_size]
        
        self._assign_bytes(ranked_morphemes)
        
        # Save mappings and model
        out_json = os.path.join(output_dir, f"{lang}_myte_{vocab_size}.json")
        out_bin = os.path.join(output_dir, f"{lang}_myte_morfessor_{vocab_size}.bin")
        
        io.write_binary_model_file(out_bin, self.morfessor_model)
        
        # Store byte maps natively as hex strings for JSON serialization
        hex_map = {k: v.hex() for k, v in self.morpheme_byte_map.items()}
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({
                "morpheme_byte_map": hex_map,
                "vocab_size": vocab_size,
                "lang": lang
            }, f, indent=2, ensure_ascii=False)
            
        if os.path.exists(temp_corpus):
            os.remove(temp_corpus)
            
        print(f"MYTE tokenizer successfully trained and saved to {output_dir}")

    def load(self, model_json_path, model_bin_path):
        with open(model_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.morpheme_byte_map = {k: bytes.fromhex(v) for k, v in data["morpheme_byte_map"].items()}
        self.byte_morpheme_map = {v: k for k, v in self.morpheme_byte_map.items()}
        
        io = morfessor.MorfessorIO()
        self.morfessor_model = io.read_binary_model_file(model_bin_path)

    def encode(self, text):
        norm_text = self._normalize(text)
        
        # We need to segment spaces and words
        # Morfessor usually segments words, so we split by whitespace to keep them intact
        words = []
        current_word = []
        
        for char in norm_text:
            if char.isspace():
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
                words.append(char)
            else:
                current_word.append(char)
        if current_word:
            words.append("".join(current_word))
            
        segments = []
        for word in words:
            if word.isspace():
                segments.append(word)
            else:
                morphemes = self.morfessor_model.viterbi_segment(word)[0]
                segments.extend(morphemes)
                
        # Now map to MYTE bytes
        encoded_bytes = bytearray()
        for seg in segments:
            if seg == self.capitalization_marker_char:
                encoded_bytes.extend(self.capitalization_marker_byte)
            elif seg in self.morpheme_byte_map:
                encoded_bytes.extend(self.morpheme_byte_map[seg])
            else:
                # Fallback to UTF-8
                encoded_bytes.extend(seg.encode('utf-8'))
                
        return encoded_bytes
    
    def get_segments(self, text):
        norm_text = self._normalize(text)
        
        words = []
        current_word = []
        
        for char in norm_text:
            if char.isspace():
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
                words.append(char)
            else:
                current_word.append(char)
        if current_word:
            words.append("".join(current_word))
            
        res_words = []
        current_res_word = []
        
        for word in words:
            if word.isspace():
                if current_res_word:
                    res_words.append(current_res_word)
                current_res_word = [] # space characters are skipped in the output segments
            else:
                morphemes = self.morfessor_model.viterbi_segment(word)[0]
                
                # Decode capitalization
                decoded_morphemes = []
                capitalize_next = False
                
                for m in morphemes:
                    if m == self.capitalization_marker_char:
                        capitalize_next = True
                        continue
                        
                    parts = m.split(self.capitalization_marker_char)
                    decoded_m = ""
                    
                    if capitalize_next and parts[0]:
                        decoded_m += parts[0][0].upper() + parts[0][1:]
                        capitalize_next = False
                    else:
                        decoded_m += parts[0]
                        
                    for part in parts[1:]:
                        if part:
                            decoded_m += part[0].upper() + part[1:]
                        else:
                            capitalize_next = True
                            
                    if decoded_m:
                        decoded_morphemes.append(decoded_m)
                
                current_res_word.extend(decoded_morphemes)
                
        if current_res_word:
            res_words.append(current_res_word)
            
        return res_words
