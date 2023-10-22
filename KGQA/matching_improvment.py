from fuzzywuzzy import fuzz

# (1-修改的次数/两个句子的总字数（不去重）)*100
print(fuzz.ratio("Where is School of Nursing (SN)", "Where is SN"))

# 如果某个字符串是另外一个字符串的子集则返回100；
# (1-修改的次数/两个句子中,短句的长度)*100
print(fuzz.partial_ratio("Where is School of Nursing (SN)", "Where is SN"))

# 忽略顺序匹配  其实就是字符按照顺序排列之后算ratio
print(fuzz.token_sort_ratio("Where is School of Nursing (SN)", "Where is SN"))

# 去重并且忽略顺序
print(fuzz.token_set_ratio("Where is School of Nursing (SN)", "Where is SN"))

