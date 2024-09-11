# @ hwang258@jh.edu

import re

def extract_words(sentence):
    words = re.findall(r"\b[\w']+\b", sentence)
    return words

def levenshtein_distance(word1, word2):
    len1, len2 = len(word1), len(word2)
    # Initialize a matrix to store the edit distances and operations
    dp = [[(0, "") for _ in range(len2 + 1)] for _ in range(len1 + 1)]

    # Initialize the first row and column
    for i in range(len1 + 1):
        dp[i][0] = (i, "d" * i)
    for j in range(len2 + 1):
        dp[0][j] = (j, "i" * j)

    # Fill in the rest of the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            # Minimum of deletion, insertion, or substitution
            deletion = dp[i - 1][j][0] + 1
            insertion = dp[i][j - 1][0] + 1
            substitution = dp[i - 1][j - 1][0] + cost
            min_dist = min(deletion, insertion, substitution)

            # Determine which operation led to the minimum distance
            if min_dist == deletion:
                operation = dp[i - 1][j][1] + "d"
            elif min_dist == insertion:
                operation = dp[i][j - 1][1] + "i"
            else:
                operation = dp[i - 1][j - 1][1] + ("s" if cost else "=")

            dp[i][j] = (min_dist, operation)

    # Backtrack to find the operations and positions
    i, j = len1, len2
    positions = []

    while i > 0 and j > 0:
        if dp[i][j][1][-1] == "d":
            positions.append((i - 1, i, 'd'))
            i -= 1
        elif dp[i][j][1][-1] == "i":
            positions.append((i, i, 'i'))
            j -= 1
        else:
            if dp[i][j][1][-1] == "s":
                positions.append((i - 1, i, 's'))
            i -= 1
            j -= 1

    while i > 0:
        positions.append((i - 1, i, 'd'))
        i -= 1

    while j > 0:
        positions.append((i, i, 'i'))
        j -= 1

    return dp[len1][len2][0], dp[len1][len2][1], positions[::-1]

def extract_spans(positions, orig_len):
    spans = []
    if not positions:
        return spans

    current_start, current_end, current_op = positions[0]
    
    for pos in positions[1:]:
        start, end, op = pos
        if op == current_op and (start == current_end or start == current_end + 1):
            current_end = end
        else:
            spans.append((current_start, current_end))
            current_start, current_end, current_op = start, end, op

    spans.append((current_start, current_end))
    
    # Handle insertions at the end
    if spans[-1][0] >= orig_len:
        spans[-1] = (orig_len, orig_len)

    return spans

def combine_nearby_spans(spans):
    if not spans:
        return spans

    combined_spans = [spans[0]]
    for current_span in spans[1:]:
        last_span = combined_spans[-1]
        if last_span[1] + 1 >= current_span[0]:  # Check if spans are adjacent or overlap
            combined_spans[-1] = [last_span[0], max(last_span[1], current_span[1])]
        else:
            combined_spans.append(current_span)
    return combined_spans

def parse_edit_zh(orig_transcript, trgt_transcript):
    word1 = extract_words(orig_transcript)
    word2 = extract_words(trgt_transcript)
    distance, operations, positions = levenshtein_distance(orig_transcript, trgt_transcript)
    spans = extract_spans(positions, len(orig_transcript))
    spans = combine_nearby_spans(spans)
    return operations, spans

def parse_tts_zh(orig_transcript, trgt_transcript):
    word1 = extract_words(orig_transcript)
    word2 = extract_words(trgt_transcript)
    distance, operations, positions = levenshtein_distance(orig_transcript, trgt_transcript)
    spans = extract_spans(positions, len(orig_transcript))
    spans = [[spans[0][0], len(orig_transcript)]]
    return spans


